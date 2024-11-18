import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from typing import Tuple, Optional
from torch.utils.data import DataLoader
from typing import Dict, Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape and project
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        out = self.proj(out)
        
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        x = x + self.attention(self.norm1(x), mask)
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, 
                 latent_size: int,
                 hidden_size: int,
                 num_layers: int = 6,
                 num_heads: int = 8):
        super().__init__()
        
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        
        # Temporal position encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, hidden_size))
        
        # Input projection
        self.input_proj = nn.Linear(latent_size, hidden_size)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        
    def forward(self, latent_sequence: torch.Tensor, mask=None):
        # Project input: [B, T, latent] -> [B, T, hidden]
        x = self.input_proj(latent_sequence)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
            
        return x

class ActorCritic(nn.Module):
    def __init__(self, 
                 latent_size: int = 32,
                 hidden_size: int = 256,
                 action_size: int = 1,
                 num_layers: int = 6):
        super().__init__()
        
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        
        # Spatio-temporal transformer for processing sequences
        self.transformer = SpatioTemporalTransformer(
            latent_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        
        # Actor head (outputs mean and log_std for continuous actions)
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_log_std = nn.Linear(hidden_size, action_size)
        
        # Critic head (outputs state value)
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, latent_sequence: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        # Process sequence through transformer
        features = self.transformer(latent_sequence)
        
        # Use the last timestep features for actor and critic
        last_features = features[:, -1, :]
        
        # Actor outputs
        action_mean = self.actor_mean(last_features)
        action_log_std = self.actor_log_std(last_features)
        action_std = torch.exp(torch.clamp(action_log_std, -20, 2))
        action_dist = Normal(action_mean, action_std)
        
        # Critic
        value = self.critic(last_features)
        
        return action_dist, value

class WorldModelEnv:
    """Simulated environment using the trained world model"""
    def __init__(self, vae, mdn_rnn, device='cuda'):
        self.vae = vae
        self.mdn_rnn = mdn_rnn
        self.device = device
        self.current_latent = None
        self.hidden_state = None
        
    def reset(self, initial_obs):
        with torch.no_grad():
            # Encode initial observation
            initial_obs = torch.FloatTensor(initial_obs).unsqueeze(0).to(self.device)
            self.current_latent, _ = self.vae.encode(initial_obs)
            self.hidden_state = None
        return self.current_latent.squeeze(0)
    
    def step(self, action):
        with torch.no_grad():
            # Prepare action
            action_tensor = torch.FloatTensor([action]).to(self.device)
            
            # Predict next latent state using MDN-RNN
            pi, mu, sigma, self.hidden_state = self.mdn_rnn(
                self.current_latent.unsqueeze(0),
                action_tensor.unsqueeze(0),
                self.hidden_state
            )
            
            # Sample next latent state from mixture
            next_latent = sample_from_mixture(pi, mu, sigma)
            self.current_latent = next_latent.squeeze(0)
            
            # Decode for visualization if needed
            next_obs = self.vae.decode(next_latent)
            
            # Simple reward function based on latent state (can be replaced with learned reward)
            reward = compute_reward(next_latent)
            
            return next_obs, reward

def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.995,
    lambda_: float = 0.95
) -> torch.Tensor:
    """Compute lambda returns for TD(lambda) learning"""
    next_values = torch.cat([values[:, 1:], values[:, -1:]], dim=1)
    inputs = rewards + gamma * (1 - dones) * next_values * (1 - lambda_)
    last = values[:, -1]
    
    returns = []
    for t in range(rewards.shape[1] - 1, -1, -1):
        last = inputs[:, t] + gamma * lambda_ * (1 - dones[:, t]) * last
        returns.append(last)
    returns = torch.stack(returns[::-1], dim=1)
    return returns


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Split heads
        qkv = self.qkv(x).chunk(3, dim=-1)  # [B, T, C] -> 3 x [B, T, C]
        q, k, v = map(lambda t: t.view(B, T, self.num_heads, self.head_size).transpose(1, 2), qkv)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        
        # Combine heads
        out = torch.matmul(attn, v)  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class WorldModelTransformer(nn.Module):
    def __init__(
        self,
        latent_size: int = 32,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_length: int = 32,
        action_size: int = 1
    ):
        super().__init__()
        
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        
        # Input embeddings
        self.latent_embed = nn.Linear(latent_size, hidden_size)
        self.action_embed = nn.Linear(action_size, hidden_size)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_length, hidden_size))
        
        # Transformer blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output heads for next state, reward, and done prediction
        self.state_pred = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, latent_size)
        )
        self.reward_pred = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)
        )
        self.done_pred = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        latents: torch.Tensor,  # [B, T, latent_size]
        actions: torch.Tensor,  # [B, T, action_size]
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Embed inputs
        h_latent = self.latent_embed(latents)
        h_action = self.action_embed(actions)
        h = h_latent + h_action
        
        # Add positional embeddings
        h = h + self.pos_embedding[:, :h.size(1)]
        
        # Apply transformer blocks
        for block in self.transformer:
            h = block(h, attention_mask)
        
        # Predict next state, reward, and done
        next_state = self.state_pred(h)
        reward = self.reward_pred(h)
        done = self.done_pred(h)
        
        return next_state, reward, done

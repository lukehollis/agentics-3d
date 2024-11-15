import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Optional

class ActorCritic(nn.Module):
    def __init__(self, latent_size: int = 32, hidden_size: int = 256, action_size: int = 1):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head (outputs mean and log_std for continuous actions)
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_log_std = nn.Linear(hidden_size, action_size)
        
        # Critic head (outputs state value)
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, latent_state: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        shared_features = self.shared(latent_state)
        
        # Actor
        action_mean = self.actor_mean(shared_features)
        action_log_std = self.actor_log_std(shared_features)
        action_std = torch.exp(torch.clamp(action_log_std, -20, 2))
        action_dist = Normal(action_mean, action_std)
        
        # Critic
        value = self.critic(shared_features)
        
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

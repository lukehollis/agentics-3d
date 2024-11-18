import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ActorCritic(nn.Module):
    def __init__(
        self,
        latent_size: int,
        hidden_size: int,
        action_size: int,
        sequence_length: int = 32,
        burn_in_length: int = 20,
        gamma: float = 0.995,
        lambda_: float = 0.95,
        entropy_weight: float = 0.001
    ):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_weight = entropy_weight
        
        # Shared encoder for both actor and critic
        self.encoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.hidden = None
    
    def reset(self, batch_size: int = 1):
        """Reset hidden states"""
        self.hidden = None
    
    def forward(
        self, 
        latent: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action distribution and value"""
        features = self.encoder(latent)
        
        # Get action logits and value
        action_logits = self.actor(features) / temperature
        value = self.critic(features)
        
        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            # Sample from action distribution
            action = torch.distributions.Categorical(
                logits=action_logits
            ).sample()
        
        return action, value
    
    def compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute lambda returns for training"""
        # Following IRIS lines 859-866
        last_value = last_value if last_value is not None else values[:, -1]
        next_values = torch.cat([values[:, 1:], last_value.unsqueeze(1)], dim=1)
        
        lambda_returns = []
        last_lambda = last_value
        
        for t in reversed(range(rewards.shape[1])):
            bootstrap = (1 - dones[:, t]) * next_values[:, t]
            lambda_return = rewards[:, t] + self.gamma * (
                (1 - self.lambda_) * bootstrap + 
                self.lambda_ * last_lambda
            )
            lambda_returns.insert(0, lambda_return)
            last_lambda = lambda_return
            
        return torch.stack(lambda_returns, dim=1)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import wandb
from tqdm import tqdm

class WorldModelTrainer:
    def __init__(
        self,
        vae,
        world_model,
        config: dict,
        device: str = 'cuda'
    ):
        self.vae = vae.to(device)
        self.world_model = world_model.to(device)
        self.device = device
        self.config = config
        
        # Optimizers
        self.vae_optimizer = torch.optim.Adam(
            self.vae.parameters(), 
            lr=config['learning_rate']
        )
        self.wm_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=config['learning_rate']
        )
    
    def train_epoch(self, train_loader: DataLoader):
        self.vae.train()
        self.world_model.train()
        
        for batch in train_loader:
            # VAE training
            obs = batch['observations'].to(self.device)
            batch_size, seq_len = obs.shape[:2]
            
            obs_flat = obs.view(-1, *obs.shape[2:])
            recon, mu, log_var = self.vae(obs_flat)
            
            # VAE losses
            recon_loss = F.mse_loss(recon, obs_flat)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            vae_loss = recon_loss + self.config['beta'] * kl_loss
            
            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.config['max_grad_norm'])
            self.vae_optimizer.step()
            
            # World Model training
            with torch.no_grad():
                z = self.vae.encode(obs_flat)[0]
                z = z.view(batch_size, seq_len, -1)
            
            actions = batch['actions'].to(self.device)
            rewards = batch['rewards'].to(self.device)
            dones = batch['dones'].to(self.device)
            
            # Create causal attention mask
            attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            attention_mask = attention_mask.to(self.device)
            
            # Predict next states, rewards, and dones
            pred_latents, pred_rewards, pred_dones = self.world_model(
                z[:, :-1], 
                actions[:, :-1],
                attention_mask
            )
            
            # World Model losses
            latent_loss = F.mse_loss(pred_latents, z[:, 1:])
            reward_loss = F.mse_loss(pred_rewards, rewards[:, :-1])
            done_loss = F.binary_cross_entropy(pred_dones, dones[:, :-1])
            
            world_model_loss = (
                latent_loss + 
                self.config['reward_scale'] * reward_loss + 
                self.config['done_scale'] * done_loss
            )
            
            self.wm_optimizer.zero_grad()
            world_model_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config['max_grad_norm'])
            self.wm_optimizer.step()
            
            if wandb.run:
                wandb.log({
                    'vae_loss': vae_loss.item(),
                    'recon_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item(),
                    'world_model_loss': world_model_loss.item(),
                    'latent_loss': latent_loss.item(),
                    'reward_loss': reward_loss.item(),
                    'done_loss': done_loss.item()
                })
    
    def validate_imagination(self, initial_obs: torch.Tensor, horizon: int = 50):
        """Validate world model by generating imagined trajectories"""
        self.vae.eval()
        self.world_model.eval()
        
        with torch.no_grad():
            # Initial latent state
            z = self.vae.encode(initial_obs)[0]
            
            # Generate trajectory
            imagined_latents = [z]
            imagined_rewards = []
            imagined_dones = []
            
            for _ in range(horizon):
                # Sample random action (for validation)
                action = torch.randint(0, self.config['action_size'], (z.size(0),))
                action = action.to(self.device)
                
                # Predict next state
                z_next, reward, done = self.world_model(
                    torch.stack(imagined_latents, dim=1),
                    action.unsqueeze(1)
                )
                
                imagined_latents.append(z_next[:, -1])
                imagined_rewards.append(reward[:, -1])
                imagined_dones.append(done[:, -1])
                
                if done.any():
                    break
            
            # Decode imagined trajectory
            imagined_obs = torch.stack([
                self.vae.decode(z) for z in imagined_latents
            ], dim=1)
            
            return imagined_obs, imagined_rewards, imagined_dones
"""Evaluation script for trained models"""

import argparse
import yaml
from pathlib import Path
from unity_env import UnityEnv
from models.vae import VAE
from models.world_model import WorldModelTransformer
from models.actor_critic import ActorCritic
import torch
import numpy as np
from typing import Dict, Optional
import wandb
from tqdm import tqdm

from collect import DataCollector
from models.vae import VAE
from models.world_model import WorldModelTransformer
from models.actor_critic import ActorCritic

class WorldModelEvaluator:
    def __init__(
        self,
        env_path: str,
        save_dir: Path,
        device: str = 'cuda',
        num_eval_episodes: int = 16,  # IRIS default
        temperature: float = 0.5      # IRIS eval temperature
    ):
        self.collector = DataCollector(
            env_path=env_path,
            save_dir=save_dir,
            sequence_length=32,
            burn_in_length=20
        )
        self.device = device
        self.num_eval_episodes = num_eval_episodes
        self.temperature = temperature
    
    @torch.no_grad()
    def evaluate_agent(
        self,
        vae: VAE,
        world_model: WorldModelTransformer,
        actor_critic: ActorCritic,
        epoch: int
    ) -> Dict[str, float]:
        """Evaluate agent in real environment"""
        vae.eval()
        world_model.eval()
        actor_critic.eval()
        
        metrics = self.collector.collect_episodes(
            num_episodes=self.num_eval_episodes,
            policy=actor_critic,
            epsilon=0.0  # No exploration during eval
        )
        
        # Log evaluation metrics
        if wandb.run:
            wandb.log({
                'eval/mean_reward': np.mean(metrics['episode_rewards']),
                'eval/mean_length': np.mean(metrics['episode_lengths']),
                'eval/total_steps': metrics['total_steps'],
                'epoch': epoch
            })
        
        return metrics
    
    @torch.no_grad()
    def evaluate_world_model(
        self,
        vae: VAE,
        world_model: WorldModelTransformer,
        initial_obs: torch.Tensor,
        horizon: int = 50
    ) -> Dict[str, torch.Tensor]:
        """Evaluate world model predictions"""
        vae.eval()
        world_model.eval()
        
        # Get initial latent state
        initial_latent = vae.encode(initial_obs.to(self.device))[0]
        
        # Generate imagined trajectory
        imagined_latents = [initial_latent]
        imagined_rewards = []
        imagined_dones = []
        
        for _ in range(horizon):
            # Sample random actions for evaluation
            action = torch.randint(
                0,
                world_model.action_size,
                (initial_latent.size(0),),
                device=self.device
            )
            
            # Predict next state
            next_latent, reward, done = world_model(
                torch.stack(imagined_latents, dim=1),
                action.unsqueeze(1)
            )
            
            imagined_latents.append(next_latent[:, -1])
            imagined_rewards.append(reward[:, -1])
            imagined_dones.append(done[:, -1])
            
            if done.any():
                break
        
        # Decode imagined observations
        imagined_obs = torch.stack([
            vae.decode(z) for z in imagined_latents
        ], dim=1)
        
        return {
            'observations': imagined_obs,
            'rewards': torch.stack(imagined_rewards, dim=1),
            'dones': torch.stack(imagined_dones, dim=1)
        }

def visualize_predictions(
    real_obs: torch.Tensor,
    predicted_obs: torch.Tensor,
    save_path: Optional[Path] = None
):
    """Visualize real vs predicted observations"""
    # Implementation of visualization logic
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/eval.yaml')
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load models
    vae = VAE.load(args.checkpoint_dir / 'vae.pt')
    world_model = WorldModelTransformer.load(args.checkpoint_dir / 'world_model.pt')
    policy = ActorCritic.load(args.checkpoint_dir / 'policy.pt')
    
    # Setup environment
    env = UnityEnv(config['env_path'])
    
    # Run evaluation episodes
    for episode in range(config['num_episodes']):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Your evaluation logic here
            with torch.no_grad():
                action = policy.act(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        
        print(f"Episode {episode}: {total_reward}")

if __name__ == '__main__':
    main()
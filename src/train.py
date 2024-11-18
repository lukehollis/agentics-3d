"""Main training script that coordinates different training phases"""

import hydra
from omegaconf import DictConfig
import wandb
from pathlib import Path
from typing import Dict, Optional

from models.vae import VAE
from models.world_model import WorldModelTransformer
from models.actor_critic import ActorCritic
from datasets import UnityDataModule
from collect import DataCollector
from utils import setup_wandb, create_experiment_dir

class WorldModelTrainer:
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = config.common.device
        
        # Initialize components
        self.vae = VAE(**config.vae).to(self.device)
        self.world_model = WorldModelTransformer(**config.world_model).to(self.device)
        self.actor_critic = ActorCritic(**config.actor_critic).to(self.device)
        
        # Setup data and collection
        self.data_module = UnityDataModule(**config.datasets)
        self.collector = DataCollector(
            env_path=config.env.path,
            save_dir=Path(config.collection.save_dir),
            sequence_length=config.common.sequence_length,
            burn_in_length=config.training.actor_critic.burn_in
        )
        
        # Initialize optimizers
        self.setup_optimizers()
        
    def train(self):
        """Main training loop following IRIS's three-step approach"""
        for epoch in range(self.config.common.epochs):
            # 1. Collect experience
            if epoch < self.config.collection.train.stop_after_epochs:
                self.collect_experience(epoch)
            
            # 2. Update world model
            if epoch > self.config.training.world_model.start_after_epochs:
                self.train_world_model(epoch)
            
            # 3. Update behavior in imagination
            if epoch > self.config.training.actor_critic.start_after_epochs:
                self.train_in_imagination(epoch)
            
            # Evaluation
            if self.config.evaluation.should and epoch % self.config.evaluation.every == 0:
                self.evaluate(epoch)
            
            # Checkpointing
            if self.config.common.do_checkpoint:
                self.save_checkpoint(epoch)
    
    def collect_experience(self, epoch: int):
        """Step 1: Collect real environment experience"""
        metrics = self.collector.collect_episodes(
            num_episodes=self.config.collection.train.num_episodes,
            policy=self.actor_critic if epoch > self.config.training.actor_critic.start_after_epochs else None,
            epsilon=self.config.collection.train.config.epsilon
        )
        if wandb.run:
            wandb.log({f'collection/{k}': v for k, v in metrics.items()})
    
    def train_world_model(self, epoch: int):
        """Step 2: Train world model on collected experience"""
        self.vae.train()
        self.world_model.train()
        
        for _ in range(self.config.training.world_model.steps_per_epoch):
            batch = next(iter(self.data_module.train_dataloader()))
            # World model training logic here
            
    def train_in_imagination(self, epoch: int):
        """Step 3: Train policy in imagined environment"""
        self.actor_critic.train()
        
        for _ in range(self.config.training.actor_critic.steps_per_epoch):
            # Sample initial states from dataset
            initial_obs = self.data_module.sample_initial_states(
                self.config.training.actor_critic.batch_num_samples
            )
            
            # Generate imagined trajectories and train actor-critic
            # Imagination training logic here

@hydra.main(config_path="config", config_name="trainer")
def main(config: DictConfig):
    setup_wandb(
        project=config.wandb.project,
        experiment_name=config.wandb.name,
        config=config
    )
    
    trainer = WorldModelTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
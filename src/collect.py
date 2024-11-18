from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
import numpy as np
import cv2
from pathlib import Path
import h5py
from typing import Dict, Optional
import torch
import wandb
from tqdm import tqdm

class DataCollector:
    def __init__(
        self,
        env_path: str,
        save_dir: Path,
        sequence_length: int = 32,
        burn_in_length: int = 20,
        image_size: tuple = (64, 64)
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.image_size = image_size
        
        # Initialize Unity environment
        self.unity_env = UnityEnvironment(
            file_name=env_path,
            seed=42,
            side_channels=[]
        )
        self.env = UnityToGymWrapper(
            self.unity_env,
            uint8_visual=True,
            allow_multiple_obs=True
        )
        
    def collect_episodes(
        self,
        num_episodes: int,
        policy: Optional[torch.nn.Module] = None,
        epsilon: float = 0.01
    ) -> Dict[str, float]:
        """Collect episodes using given policy or random actions"""
        metrics = {
            'total_steps': 0,
            'total_reward': 0,
            'episode_lengths': [],
            'episode_rewards': []
        }
        
        for episode in tqdm(range(num_episodes), desc="Collecting episodes"):
            episode_path = self.save_dir / f"episode_{episode:06d}.h5"
            
            with h5py.File(episode_path, 'w') as f:
                # Initialize episode buffers
                observations = []
                actions = []
                rewards = []
                dones = []
                
                obs = self.env.reset()
                obs = self.preprocess_observation(obs)
                episode_reward = 0
                
                while True:
                    observations.append(obs)
                    
                    # Get action from policy or random
                    if policy is not None and np.random.random() > epsilon:
                        with torch.no_grad():
                            action = policy(
                                torch.FloatTensor(obs).unsqueeze(0).to(policy.device)
                            ).cpu().numpy()[0]
                    else:
                        action = self.env.action_space.sample()
                    
                    actions.append(action)
                    
                    # Take step in environment
                    next_obs, reward, done, info = self.env.step(action)
                    next_obs = self.preprocess_observation(next_obs)
                    
                    rewards.append(reward)
                    dones.append(done)
                    episode_reward += reward
                    
                    if done:
                        break
                        
                    obs = next_obs
                
                # Save episode data
                f.create_dataset('observations', data=np.array(observations))
                f.create_dataset('actions', data=np.array(actions))
                f.create_dataset('rewards', data=np.array(rewards))
                f.create_dataset('dones', data=np.array(dones))
                
                # Save episode metadata
                f.attrs['total_reward'] = episode_reward
                f.attrs['length'] = len(observations)
                f.attrs['success'] = info.get('success', False)
                
                # Update metrics
                metrics['total_steps'] += len(observations)
                metrics['total_reward'] += episode_reward
                metrics['episode_lengths'].append(len(observations))
                metrics['episode_rewards'].append(episode_reward)
                
                if wandb.run:
                    wandb.log({
                        'episode_reward': episode_reward,
                        'episode_length': len(observations)
                    })
        
        return metrics
    
    def preprocess_observation(self, obs):
        """Preprocess observation following World Models specs"""
        if isinstance(obs, dict):
            obs = obs.get('visual', obs.get('rgb', obs))
        obs = cv2.resize(obs, self.image_size)
        return obs.astype(np.float32) / 255.0

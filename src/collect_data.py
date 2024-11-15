from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
import numpy as np
import cv2
from pathlib import Path
import torch
from torch.utils.data import Dataset

class WorldModelDataset(Dataset):
    def __init__(self, path: Path, sequence_length: int = 6):
        self.path = path
        self.sequence_length = sequence_length
        self.episodes = []
        self.total_steps = 0
        
    def add_episode(self, observations, actions, rewards, dones):
        episode = {
            'observations': observations,
            'actions': actions, 
            'rewards': rewards,
            'dones': dones
        }
        self.episodes.append(episode)
        self.total_steps += len(observations)

    def save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'episodes': self.episodes,
            'total_steps': self.total_steps,
            'sequence_length': self.sequence_length
        }, self.path / 'dataset.pt')

def preprocess_observation(obs, target_size=(64, 64)):
    """Preprocess observation following Diamond/World Models specs"""
    if isinstance(obs, dict):
        obs = obs.get('visual', obs.get('rgb', obs))
    obs = cv2.resize(obs, target_size)
    return obs.astype(np.float32) / 255.0

def collect_game_data(num_episodes=1000, sequence_length=6):
    """Collect data following Diamond/World Models approach"""
    unity_env = UnityEnvironment(seed=1)
    env = UnityToGymWrapper(
        unity_env,
        uint8_visual=True,
        allow_multiple_obs=True
    )
    
    # Create train/test datasets
    train_data = WorldModelDataset(Path("dataset/train"), sequence_length)
    test_data = WorldModelDataset(Path("dataset/test"), sequence_length)
    
    for episode in range(num_episodes):
        observations = []
        actions = []
        rewards = []
        dones = []
        
        obs = env.reset()
        obs = preprocess_observation(obs)
        done = False
        
        while not done:
            observations.append(obs)
            
            # Sample action (can be replaced with expert policy)
            action = env.action_space.sample()
            actions.append(action)
            
            next_obs, reward, done, info = env.step(action)
            next_obs = preprocess_observation(next_obs)
            
            rewards.append(reward)
            dones.append(done)
            obs = next_obs
            
        # Add to appropriate dataset (80/20 split)
        if episode < num_episodes * 0.8:
            train_data.add_episode(
                np.array(observations),
                np.array(actions),
                np.array(rewards),
                np.array(dones)
            )
        else:
            test_data.add_episode(
                np.array(observations),
                np.array(actions),
                np.array(rewards),
                np.array(dones)
            )
    
    # Save datasets
    train_data.save()
    test_data.save()
    
    return train_data, test_data

# Save collected data
def save_dataset(dataset, filename):
    np.save(filename, dataset)

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import h5py
import torch.nn.functional as F

class Episode:
    """Container for episode data"""
    def __init__(self, path: Path):
        self.path = path
        with h5py.File(path, 'r') as f:
            self.length = len(f['observations'])
            # Cache episode metadata but not data itself
            self.total_reward = float(f['rewards'][:].sum())
            self.has_success = bool(f.attrs.get('success', False))
    
    def load_sequence(self, start_idx: int, sequence_length: int) -> Dict[str, torch.Tensor]:
        """Load a sequence from the episode"""
        with h5py.File(self.path, 'r') as f:
            obs = torch.from_numpy(
                f['observations'][start_idx:start_idx + sequence_length]
            ).float()
            actions = torch.from_numpy(
                f['actions'][start_idx:start_idx + sequence_length]
            ).float()
            rewards = torch.from_numpy(
                f['rewards'][start_idx:start_idx + sequence_length]
            ).float()
            dones = torch.from_numpy(
                f['dones'][start_idx:start_idx + sequence_length]
            ).float() if 'dones' in f else torch.zeros(sequence_length)
            
            return {
                'observations': obs,
                'actions': actions,
                'rewards': rewards,
                'dones': dones
            }

class UnityEpisodeDataset(Dataset):
    """Dataset for loading and processing Unity episode data"""
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 32,
        transform=None,
        train: bool = True,
        burn_in_length: int = 20  # Following IRIS
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        self.train = train
        self.burn_in_length = burn_in_length
        
        self.episodes: List[Episode] = []
        self.episode_index: List[Tuple[int, int]] = []
        self._load_episodes()
    
    def _load_episodes(self):
        """Load episode metadata and create index"""
        for episode_path in sorted(self.data_dir.glob("*.h5")):
            episode = Episode(episode_path)
            self.episodes.append(episode)
            
            # Create sliding windows including burn-in context
            total_length = self.sequence_length + self.burn_in_length
            for start_idx in range(0, episode.length - total_length + 1):
                self.episode_index.append((len(self.episodes) - 1, start_idx))
    
    def __len__(self) -> int:
        return len(self.episode_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_idx, start_idx = self.episode_index[idx]
        episode = self.episodes[episode_idx]
        
        # Load sequence with burn-in context
        sequence = episode.load_sequence(
            start_idx,
            self.sequence_length + self.burn_in_length
        )
        
        if self.transform:
            sequence['observations'] = self.transform(sequence['observations'])
        
        # Split into burn-in and main sequence
        result = {}
        for key, value in sequence.items():
            result[f'burn_in_{key}'] = value[:self.burn_in_length]
            result[key] = value[self.burn_in_length:]
        
        return result

class UnityDataTransform:
    """Transform class for Unity observations"""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (64, 64),
        normalize: bool = True
    ):
        self.image_size = image_size
        self.normalize = normalize
    
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-2:] != self.image_size:
            obs = F.interpolate(
                obs,
                size=self.image_size,
                mode='bilinear',
                align_corners=False
            )
        
        if self.normalize:
            obs = obs / 255.0
        
        return obs

class UnityDataModule:
    """Data module to handle all data operations"""
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 32,
        batch_size: int = 64,  # Match IRIS batch size
        num_workers: int = 4,
        train_val_split: float = 0.9,
        burn_in_length: int = 20
    ):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.burn_in_length = burn_in_length
        
        self.transform = UnityDataTransform()
    
    def setup(self):
        """Setup train and validation datasets"""
        full_dataset = UnityEpisodeDataset(
            self.data_dir,
            self.sequence_length,
            transform=self.transform,
            train=True,
            burn_in_length=self.burn_in_length
        )
        
        # Split into train/val
        train_size = int(len(full_dataset) * self.train_val_split)
        val_size = len(full_dataset) - train_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
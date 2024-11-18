import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import yaml
import wandb
from einops import rearrange

def load_config(config_path: Union[str, Path]) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_observation(
    obs: Union[np.ndarray, torch.Tensor],
    image_size: Tuple[int, int] = (64, 64)
) -> torch.Tensor:
    """Preprocess observation for model input"""
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)
    
    # Handle different input formats
    if isinstance(obs, dict):
        obs = obs.get('visual', obs.get('rgb', obs))
    
    # Ensure BCHW format
    if obs.ndim == 3:
        obs = obs.unsqueeze(0)
    obs = rearrange(obs, 'b h w c -> b c h w')
    
    # Resize if needed
    if obs.shape[-2:] != image_size:
        obs = torch.nn.functional.interpolate(
            obs,
            size=image_size,
            mode='bilinear',
            align_corners=False
        )
    
    return obs.float() / 255.0

def create_experiment_dir(
    base_dir: Union[str, Path],
    experiment_name: str,
    config: dict
) -> Path:
    """Create experiment directory and save config"""
    base_dir = Path(base_dir)
    exp_dir = base_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    return exp_dir

def setup_wandb(
    project_name: str,
    experiment_name: str,
    config: dict,
    mode: str = 'online'
) -> None:
    """Initialize wandb logging"""
    wandb.init(
        project=project_name,
        name=experiment_name,
        config=config,
        mode=mode
    )

def compute_metrics(
    episode_data: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Compute episode metrics"""
    metrics = {
        'episode_return': float(episode_data['rewards'].sum()),
        'episode_length': len(episode_data['rewards']),
        'mean_reward': float(episode_data['rewards'].mean()),
        'success_rate': float(episode_data['rewards'].max() > 0)
    }
    return metrics

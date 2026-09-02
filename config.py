"""
Configuration file for Mario RL Agent.
Contains hyperparameters and settings for training DQN/PPO agents.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """Hyperparameters and configuration for the Mario RL agent."""

    # Environment settings
    env_name: str = 'SuperMarioBros-1-1-v0'
    action_type: str = 'simple'  # 'simple', 'rights', or 'complex'
    frame_stack: int = 4         # Number of frames to stack
    frame_size: Tuple[int, int] = (84, 84)  # Resized frame dimensions
    frame_skip: int = 4          # Number of frames to skip (action repeat)

    # Training settings
    total_timesteps: int = 10_000_000
    learning_rate: float = 0.00025
    gamma: float = 0.99          # Discount factor
    batch_size: int = 32
    buffer_size: int = 100_000   # Replay buffer size

    # Exploration settings (for DQN)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.999995  # Linear decay over timesteps
    epsilon_decay_steps: int = 1_000_000  # Steps over which to decay epsilon

    # Network settings
    target_update_freq: int = 10_000  # Steps between target network updates
    train_freq: int = 4               # Steps between training updates
    gradient_clip: float = 10.0       # Gradient clipping value

    # Checkpoint and logging
    checkpoint_freq: int = 100_000    # Steps between model checkpoints
    log_freq: int = 10_000            # Steps between tensorboard logs
    eval_freq: int = 50_000           # Steps between evaluations
    eval_episodes: int = 5            # Number of episodes for evaluation

    # Device
    device: str = 'cuda'  # 'cuda' or 'cpu'

    # Paths
    checkpoint_dir: str = 'checkpoints/'
    log_dir: str = 'logs/'
    tensorboard_dir: str = 'runs/'


# Default configuration instance
config = Config()
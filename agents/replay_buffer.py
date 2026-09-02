"""
Experience replay buffer implementation for DQN.
"""

import numpy as np
import torch
from collections import deque, namedtuple
from typing import Deque, Tuple, Optional, List
import random


# Define the experience tuple
Experience = namedtuple(
    'Experience',
    ['state', 'action', 'reward', 'next_state', 'done']
)


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    """

    """

    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer: Deque[Experience] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add an experience to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Not enough experiences in buffer. "
                f"Have {len(self.buffer)}, need {batch_size}"
            )

        batch = random.sample(self.buffer, batch_size)

        # Convert to numpy arrays first for efficiency
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch], dtype=np.uint8)

        # Convert to PyTorch tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.BoolTensor(dones)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough experiences for sampling.

        Args:
            batch size: Required batch size

        Returns:
            True if buffer has at least batch_size experiences
        """
        return len(self.buffer) >= batch_size


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer (TODO: implement later).
    For now, this is a placeholder that uses uniform sampling.
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        Initialize the prioritized replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: Deque[Experience] = deque(maxlen=capacity)
        self.priorities: Deque[float] = deque(maxlen=capacity)
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add an experience to the buffer with max priority.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray
    ]:
        """
        Sample a batch of experiences using prioritized sampling.
        Currently implements uniform sampling as placeholder.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        # For now, fall back to uniform sampling
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Not enough experiences in buffer. "
                f"Have {len(self.buffer)}, need {batch_size}"
            )

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        # Convert to arrays
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch], dtype=np.uint8)

        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.BoolTensor(dones)

        # Uniform weights (placeholder for prioritized replay)
        weights = np.ones(batch_size, dtype=np.float32)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities of sampled experiences.
        Placeholder implementation.

        Args:
            indices: Indices of experiences to update
            priorities: New priority values
        """
        # TODO: Implement priority updates
        pass

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
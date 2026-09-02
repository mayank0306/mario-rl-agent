"""
DQN Agent for Mario RL.
Includes epsilon-greedy policy, replay buffer, and learning step.
Currently implements a random agent - replace with actual DQN logic.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Tuple, Optional
from ..models.cnn import DQNNet
from .replay_buffer import ReplayBuffer, Experience
from ..config import config


class DQNAgent:
    """
    Deep Q-Network Agent for playing Super Mario Bros.

    Current implementation: Random agent (placeholder)
    TODO: Implement actual DQN learning logic
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (4, 84, 84),
        n_actions: int = 7,
        learning_rate: float = None,
        gamma: float = None,
        epsilon_start: float = None,
        epsilon_end: float = None,
        epsilon_decay: float = None,
        buffer_size: int = None,
        batch_size: int = None,
        target_update_freq: int = None,
        device: str = None
    ):
        """
        Initialize the DQN agent.

        Args:
            input_shape: Shape of input frames (channels, height, width)
            n_actions: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Starting epsilon for exploration
            epsilon_end: Final epsilon for exploration
            epsilon_decay: Epsilon decay rate
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: How often to update target network
            device: Device to run on ('cuda' or 'cpu')
        """
        # Use config values if not provided
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate or config.learning_rate
        self.gamma = gamma or config.gamma
        self.epsilon_start = epsilon_start or config.epsilon_start
        self.epsilon_end = epsilon_end or config.epsilon_end
        self.epsilon_decay = epsilon_decay or config.epsilon_decay
        self.buffer_size = buffer_size or config.buffer_size
        self.batch_size = batch_size or config.batch_size
        self.target_update_freq = target_update_freq or config.target_update_freq
        self.device = device or config.device

        # Initialize networks
        self.policy_net = DQNNet(
            input_shape=input_shape,
            n_actions=n_actions
        ).to(self.device)

        self.target_net = DQNNet(
            input_shape=input_shape,
            n_actions=n_actions
        ).to(self.device)

        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is only for inference

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=self.buffer_size)

        # Exploration tracking
        self.epsilon = self.epsilon_start
        self.steps_done = 0

        # Loss function
        self.criterion = nn.SmoothL1Loss()

    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action using epsilon-greedy policy.

        Current implementation: Random action (placeholder)
        TODO: Implement actual epsilon-greedy with Q-values

        Args:
            state: Current state observation

        Returns:
            action: Selected action index
        """
        # TODO: Replace random action with epsilon-greedy policy
        # With probability epsilon: select random action
        # With probability (1 - epsilon): select action with highest Q-value

        # For now, return random action as placeholder
        return random.randrange(self.n_actions)

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store experience in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self) -> Optional[float]:
        """
        Learn from experiences in replay buffer.

        Current implementation: Returns None (placeholder)
        TODO: Implement actual DQN learning step

        Returns:
            loss: Training loss value (if learning occurred)
        """
        # TODO: Implement DQN learning logic:
        # 1. Sample batch from replay buffer
        # 2. Compute current Q-values
        # 3. Compute target Q-values
        # 4. Compute loss
        # 5. Optimize policy network
        # 6. Update target network periodically
        # 7. Decay epsilon

        # For now, return None as placeholder
        return None

    def update_epsilon(self) -> None:
        """Update epsilon for exploration."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )

    def update_target_network(self) -> None:
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path: str) -> None:
        """
        Save the policy network to disk.

        Args:
            path: File path to save model
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
        }, path)

    def load_model(self, path: str) -> None:
        """
        Load the policy network from disk.

        Args:
            path: File path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']


if __name__ == "__main__":
    # Simple test of the agent
    agent = DQNAgent()
    print(f"Agent initialized with device: {agent.device}")
    print(f"Policy network: {agent.policy_net}")
    print(f"Epsilon: {agent.epsilon}")

    # Test action selection
    dummy_state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)
    action = agent.select_action(dummy_state)
    print(f"Selected action: {action}")

    # Test experience storage
    agent.store_experience(dummy_state, action, 1.0, dummy_state, False)
    print(f"Buffer size: {len(agent.replay_buffer)}")

    print("DQN agent test completed successfully!")
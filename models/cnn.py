"""
CNN architecture for Mario RL Agent.
Feature extractor for processing stacked grayscale frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MarioCNN(nn.Module):
    """
    Convolutional Neural Network for processing Mario game frames.
    Takes stacked grayscale frames as input and outputs features for RL agent.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (4, 84, 84),
        output_size: int = 512
    ):
        """
        Initialize the CNN.

        Args:
            input_shape: Shape of input frames (channels, height, width)
            output_size: Size of output feature vector
        """
        super(MarioCNN, self).__init__()

        self.input_shape = input_shape
        self.output_size = output_size

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1
        )

        # Calculate the size of flattened features after conv layers
        self.feature_size = self._get_conv_output(input_shape)

        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, output_size)
        self.fc2 = nn.Linear(output_size, output_size)

    def _get_conv_output(self, shape: Tuple[int, int, int]) -> int:
        """
        Calculate the size of the output from convolutional layers.

        Args:
            shape: Input shape (channels, height, width)

        Returns:
            Size of flattened conv output
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return int(torch.prod(torch.tensor(x.shape[1:])))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Feature tensor of shape (batch_size, output_size)
        """
        # Normalize pixel values to [0, 1]
        x = x.float() / 255.0

        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class DQNNet(nn.Module):
    """
    DQN Network that combines CNN feature extractor with Q-value head.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (4, 84, 84),
        n_actions: int = 7,
        output_size: int = 512
    ):
        """
        Initialize the DQN network.

        Args:
            input_shape: Shape of input frames (channels, height, width)
            n_actions: Number of possible actions
            output_size: Size of CNN feature vector
        """
        super(DQNNet, self).__init__()

        self.cnn = MarioCNN(input_shape=input_shape, output_size=output_size)
        self.q_head = nn.Linear(output_size, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Q-values tensor of shape (batch_size, n_actions)
        """
        features = self.cnn(x)
        q_values = self.q_head(features)
        return q_values


class PPONet(nn.Module):
    """
    PPO Network that combines CNN feature extractor with policy and value heads.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (4, 84, 84),
        n_actions: int = 7,
        output_size: int = 512
    ):
        """
        Initialize the PPO network.

        Args:
            input_shape: Shape of input frames (channels, height, width)
            n_actions: Number of possible actions
            output_size: Size of CNN feature vector
        """
        super(PPONet, self).__init__()

        self.cnn = MarioCNN(input_shape=input_shape, output_size=output_size)
        self.policy_head = nn.Linear(output_size, n_actions)
        self.value_head = nn.Linear(output_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tuple of (policy_logits, value)
                policy_logits: Tensor of shape (batch_size, n_actions)
                value: Tensor of shape (batch_size, 1)
        """
        features = self.cnn(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value


if __name__ == "__main__":
    # Simple test of the CNN
    cnn = MarioCNN(input_shape=(4, 84, 84))
    dummy_input = torch.randn(1, 4, 84, 84)
    output = cnn(dummy_input)
    print(f"CNN output shape: {output.shape}")

    # Test DQN net
    dqn_net = DQNNet(input_shape=(4, 84, 84), n_actions=7)
    dqn_output = dqn_net(dummy_input)
    print(f"DQN output shape: {dqn_output.shape}")

    # Test PPO net
    ppo_net = PPONet(input_shape=(4, 84, 84), n_actions=7)
    policy_logits, value = ppo_net(dummy_input)
    print(f"PPO policy shape: {policy_logits.shape}")
    print(f"PPO value shape: {value.shape}")

    print("CNN architecture test completed successfully!")
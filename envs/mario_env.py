"""
Mario environment wrapper.
Handles preprocessing: grayscale, resize, frame stacking, frame skipping, and action space simplification.
"""

import numpy as np
import gymnasium as gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import cv2
from collections import deque
from typing import Tuple, Deque, Optional
import gymnasium.spaces as spaces


class MarioEnv:
    """
    Wrapper for Super Mario Bros environment with preprocessing.

    Preprocessing steps:
    1. Convert to grayscale
    2. Resize to 84x84
    3. Frame stacking (4 frames)
    4. Frame skipping (4 frames)
    5. Action space simplification (simple movements)
    """

    def __init__(
        self,
        env_name: str = 'SuperMarioBros-1-1-v0',
        action_type: str = 'simple',
        frame_stack: int = 4,
        frame_size: Tuple[int, int] = (84, 84),
        frame_skip: int = 4
    ):
        """
        Initialize the Mario environment.

        Args:
            env_name: Name of the gym-super-mario-bros environment
            action_type: Type of action space ('simple', 'rights', 'complex')
            frame_stack: Number of frames to stack for state representation
            frame_size: Target size for resized frames (height, width)
            frame_skip: Number of frames to skip (action repeat)
        """
        self.env_name = env_name
        self.action_type = action_type
        self.frame_stack = frame_stack
        self.frame_size = frame_size
        self.frame_skip = frame_skip

        # Create base environment
        self.env = gym.make(env_name, render_mode='rgb_array')

        # Set action space based on type
        if action_type == 'simple':
            self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        elif action_type == 'rights':
            self.env = JoypadSpace(self.env, RIGHT_ONLY)
        elif action_type == 'complex':
            self.env = JoypadSpace(self.env, COMPLEX_MOVEMENT)
        else:
            raise ValueError(f"Unknown action_type: {action_type}")

        # Define observation space (stacked frames)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(frame_stack, *frame_size),
            dtype=np.uint8
        )

        # Define action space
        self.action_space = self.env.action_space

        # Frame stacking buffer
        self.frames: Deque[np.ndarray] = deque(maxlen=frame_stack)

        # Life tracking for reward shaping
        self.lives = 2  # Mario starts with 2 lives

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment and return initial state.

        Returns:
            state: Stacked frames as numpy array
            info: Additional information dictionary
        """
        obs, info = self.env.reset(seed=seed)
        self.lives = 2  # Reset life counter

        # Preprocess initial frame and fill frame stack
        processed_frame = self._preprocess_frame(obs)
        for _ in range(self.frame_stack):
            self.frames.append(processed_frame)

        return self._get_state(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take a step in the environment with frame skipping.

        Args:
            action: Action to take

        Returns:
            state: Stacked frames after step
            reward: Total reward over frame_skip steps
            done: Whether episode is finished
            truncated: Whether episode was truncated (time limit)
            info: Additional information
        """
        total_reward = 0.0
        done = False
        truncated = False
        info = {}

        for _ in range(self.frame_skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward

            # Life-based reward shaping (penalty for losing life)
            if 'life' in info and info['life'] < self.lives:
                total_reward -= 10.0  # Penalty for losing life
                self.lives = info['life']

            if done or truncated:
                break

        # Preprocess and store frame
        processed_frame = self._preprocess_frame(obs)
        self.frames.append(processed_frame)

        return self._get_state(), total_reward, done, truncated, info

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame: convert to grayscale and resize.

        Args:
            frame: RGB frame from environment

        Returns:
            Preprocessed grayscale frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize
        resized = cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)
        return resized

    def _get_state(self) -> np.ndarray:
        """
        Get current state as stacked frames.

        Returns:
            Stacked frames as numpy array with shape (frame_stack, height, width)
        """
        return np.array(self.frames, dtype=np.uint8)

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()


# Factory function for easy environment creation
def make_mario_env(
    env_name: str = 'SuperMarioBros-1-1-v0',
    action_type: str = 'simple',
    frame_stack: int = 4,
    frame_size: Tuple[int, int] = (84, 84),
    frame_skip: int = 4
) -> MarioEnv:
    """
    Factory function to create a Mario environment.

    Args:
        env_name: Name of the gym-super-mario-bros environment
        action_type: Type of action space ('simple', 'rights', 'complex')
        frame_stack: Number of frames to stack for state representation
        frame_size: Target size for resized frames (height, width)
        frame_skip: Number of frames to skip (action repeat)

    Returns:
        Configured MarioEnv instance
    """
    return MarioEnv(
        env_name=env_name,
        action_type=action_type,
        frame_stack=frame_stack,
        frame_size=frame_size,
        frame_skip=frame_skip
    )


if __name__ == "__main__":
    # Simple test of the environment
    env = make_mario_env()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"Initial state shape: {obs.shape}")

    # Take a few random steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.2f}, done={done}")
        if done or truncated:
            break

    env.close()
    print("Environment test completed successfully!")
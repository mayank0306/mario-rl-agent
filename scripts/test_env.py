"""
Quick script to verify the Mario environment works.
Renders a few random steps to test basic functionality.
"""

import time
import numpy as np
from envs.mario_env import make_mario_env


def test_env(
    env_name: str = 'SuperMarioBros-1-1-v0',
    action_type: str = 'simple',
    frame_stack: int = 4,
    frame_size: tuple = (84, 84),
    frame_skip: int = 4,
    n_steps: int = 20,
    render: bool = True,
    delay: float = 0.1
):
    """
    Test the Mario environment by taking random steps.

    Args:
        env_name: Name of the gym-super-mario-bros environment
        action_type: Type of action space ('simple', 'rights', 'complex')
        frame_stack: Number of frames to stack for state representation
        frame_size: Target size for resized frames (height, width)
        frame_skip: Number of frames to skip (action repeat)
        n_steps: Number of random steps to take
        render: Whether to render the environment
        delay: Delay between steps for visualization (seconds)
    """
    print("Testing Mario Environment")
    print("="*40)
    print(f"Environment: {env_name}")
    print(f"Action type: {action_type}")
    print(f"Frame stack: {frame_stack}")
    print(f"Frame size: {frame_size}")
    print(f"Frame skip: {frame_skip}")
    print(f"Steps to run: {n_steps}")
    print("="*40)

    # Create environment
    env = make_mario_env(
        env_name=env_name,
        action_type=action_type,
        frame_stack=frame_stack,
        frame_size=frame_size,
        frame_skip=frame_skip
    )

    # Enable rendering if requested
    if render:
        print("Rendering enabled - you should see the game window")

    # Reset environment
    print("\nResetting environment...")
    state, info = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"State dtype: {state.dtype}")
    print(f"State min/max: {state.min()}/{state.max()}")

    # Get environment info
    print(f"\nAction space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Take random steps
    print(f"\nTaking {n_steps} random steps...")
    total_reward = 0

    for step in range(n_steps):
        # Sample random action
        action = env.action_space.sample()

        # Take step
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Print step info
        print(f"Step {step+1:2d}: Action={action:2d}, Reward={reward:6.2f}, "
              f"Total Reward={total_reward:6.2f}, Done={done}, Truncated={truncated}")

        # Check if state shape is correct
        if next_state.shape != state.shape:
            print(f"ERROR: State shape changed from {state.shape} to {next_state.shape}")
            break

        # Update state
        state = next_state

        # Render if enabled
        if render:
            env.render()
            time.sleep(delay)

        # Break if episode ended
        if done or truncated:
            print(f"Episode ended after {step+1} steps")
            break

    print(f"\nTest completed!")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final state shape: {state.shape}")

    # Close environment
    env.close()
    print("Environment closed successfully!")


def test_preprocessing():
    """Test the preprocessing steps individually."""
    print("\nTesting Preprocessing Steps")
    print("="*40)

    import cv2
    import numpy as np

    # Create a dummy RGB frame (like what Mario environment would produce)
    dummy_frame = np.random.randint(0, 255, (240, 256, 3), dtype=np.uint8)
    print(f"Original frame shape: {dummy_frame.shape}")
    print(f"Original frame dtype: {dummy_frame.dtype}")

    # Test grayscale conversion
    gray = cv2.cvtColor(dummy_frame, cv2.COLOR_RGB2GRAY)
    print(f"After grayscale: {gray.shape}")

    # Test resizing
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    print(f"After resizing: {resized.shape}")

    # Test normalization (what the CNN does)
    normalized = resized.astype(np.float32) / 255.0
    print(f"After normalization: min={normalized.min():.3f}, max={normalized.max():.3f}")

    print("Preprocessing test completed!")


if __name__ == "__main__":
    # Test preprocessing first
    test_preprocessing()

    # Test environment
    print("\n" + "="*50)
    test_env(
        env_name='SuperMarioBros-1-1-v0',
        action_type='simple',
        frame_stack=4,
        frame_size=(84, 84),
        frame_skip=4,
        n_steps=20,
        render=True,
        delay=0.1
    )

    print("\nAll tests completed successfully!")
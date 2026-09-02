"""
Evaluation script for Mario RL Agent.
Loads a trained model and watches/records it play.
Currently implements random playback - replace with actual learned policy.
"""

import torch
import numpy as np
import random
import argparse
import os
from typing import Tuple, List
import time

from ..envs.mario_env import make_mario_env
from ..agents.dqn_agent import DQNAgent
from ..config import config


def evaluate(
    model_path: str,
    env_name: str = None,
    action_type: str = None,
    frame_stack: int = None,
    frame_size: Tuple[int, int] = None,
    frame_skip: int = None,
    n_episodes: int = None,
    render: bool = None,
    record_video: bool = None,
    video_path: str = None,
    fps: int = None
) -> Tuple[float, float]:
    """
    Evaluate a trained Mario RL agent.

    Current implementation: Plays random actions (placeholder)
    TODO: Replace with actual learned policy from loaded model

    Args:
        model_path: Path to the trained model checkpoint
        env_name: Name of the environment
        action_type: Type of action space
        frame_stack: Number of frames to stack
        frame_size: Size to resize frames to
        frame_skip: Number of frames to skip
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        record_video: Whether to record a video
        video_path: Path to save video
        fps: Frames per second for video

    Returns:
        Tuple of (mean_reward, mean_length)
    """
    # Use config values if not provided
    env_name = env_name or config.env_name
    action_type = action_type or config.action_type
    frame_stack = frame_stack or config.frame_stack
    frame_size = frame_size or config.frame_size
    frame_skip = frame_skip or config.frame_skip
    n_episodes = n_episodes or config.eval_episodes
    render = render if render is not None else True
    record_video = record_video or False
    video_path = video_path or "mario_gameplay.mp4"
    fps = fps or 30

    # Create environment
    render_mode = 'human' if render else 'rgb_array'
    env = make_mario_env(
        env_name=env_name,
        action_type=action_type,
        frame_stack=frame_stack,
        frame_size=frame_size,
        frame_skip=frame_skip
    )
    # Override render mode if needed
    if hasattr(env.env, 'render_mode'):
        env.env.render_mode = render_mode

    # Get environment info
    observation_space = env.observation_space
    action_space = env.action_space

    print(f"Observation space: {observation_space}")
    print(f"Action space: {action_space}")
    print(f"State shape: {observation_space.shape}")
    print(f"Number of actions: {action_space.n}")

    # Initialize agent
    agent = DQNAgent(
        input_shape=observation_space.shape,
        n_actions=action_space.n
    )

    # Load trained model
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        agent.load_model(model_path)
        # Set evaluation mode (no exploration)
        agent.epsilon = agent.epsilon_end if hasattr(agent, 'epsilon_end') else 0.05
    else:
        print(f"Warning: Model file {model_path} not found. Using random agent.")

    # Video recording setup (placeholder - would need additional libraries like cv2.VideoWriter)
    if record_video:
        print(f"Video recording to {video_path} (placeholder - actual recording not implemented)")
        # TODO: Implement actual video recording using OpenCV or similar

    # Evaluation loop
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False

        # Reset agent's internal state if needed (for RNN-based agents)
        # agent.reset()  # Uncomment if agent has reset method

        while not (done or truncated):
            # Render environment
            if render:
                env.render()
                # Small delay to make visualization watchable
                time.sleep(0.01)

            # Select action (currently random - TODO: implement actual policy)
            # TODO: Replace random action with agent.select_action(state) in evaluation mode
            action = agent.select_action(state)  # This still uses epsilon-greedy
            # For pure evaluation, we might want to set epsilon=0 or use a different method

            # Take step in environment
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            # Store next state
            state = next_state

            # Optional: break if episode gets too long
            if episode_length > 5000:  # Prevent infinite loops
                print("Warning: Episode too long, terminating")
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {episode + 1}/{n_episodes}: "
              f"Reward: {episode_reward:.2f}, Length: {episode_length}")

        # Optional: delay between episodes
        if episode < n_episodes - 1:
            time.sleep(1.0)

    # Close environment
    env.close()

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    print("\n" + "="*50)
    print(f"Evaluation Results ({n_episodes} episodes)")
    print("="*50)
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Length: {mean_length:.2f} ± {std_length:.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print("="*50)

    return mean_reward, mean_length


def record_gameplay(
    model_path: str,
    env_name: str = None,
    action_type: str = None,
    frame_stack: int = None,
    frame_size: Tuple[int, int] = None,
    frame_skip: int = None,
    output_path: str = None,
    max_steps: int = None,
    fps: int = None
):
    """
    Record gameplay video of the trained agent.

    Current implementation: Placeholder - actual recording not implemented
    TODO: Implement actual video recording using OpenCV

    Args:
        model_path: Path to the trained model checkpoint
        env_name: Name of the environment
        action_type: Type of action space
        frame_stack: Number of frames to stack
        frame_size: Size to resize frames to
        frame_skip: Number of frames to skip
        output_path: Path to save video
        max_steps: Maximum steps to record
        fps: Frames per second for video
    """
    # Use config values if not provided
    env_name = env_name or config.env_name
    action_type = action_type or config.action_type
    frame_stack = frame_stack or config.frame_stack
    frame_size = frame_size or config.frame_size
    frame_skip = frame_skip or config.frame_skip
    output_path = output_path or "mario_gameplay.mp4"
    max_steps = max_steps or 5000
    fps = fps or 30

    print(f"Recording gameplay to {output_path}")
    print("NOTE: Actual video recording not implemented in this placeholder version")
    print("To implement, you would need to:")
    print("1. Capture frames from env.render()")
    print("2. Use OpenCV VideoWriter to save frames as video")
    print("3. Set appropriate codec and FPS")

    # Create environment
    env = make_mario_env(
        env_name=env_name,
        action_type=action_type,
        frame_stack=frame_stack,
        frame_size=frame_size,
        frame_skip=frame_skip
    )

    # Load model
    agent = DQNAgent(
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n
    )

    if os.path.exists(model_path):
        agent.load_model(model_path)
        agent.epsilon = 0.01  # Low exploration for recording
    else:
        print(f"Warning: Model file {model_path} not found. Using random agent.")

    # TODO: Actual video recording implementation would go here
    # For now, just run the agent without recording
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0

    for step in range(max_steps):
        # env.render()  # Would capture this frame for video
        action = agent.select_action(state)  # Placeholder - should use learned policy
        next_state, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1
        state = next_state

        if done or truncated:
            break

    print(f"Recorded gameplay: Reward: {episode_reward:.2f}, Length: {episode_length}")
    env.close()


def main():
    """Parse command line arguments and start evaluation."""
    parser = argparse.ArgumentDescription("Evaluate a trained Mario RL agent")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--env", type=str, default=None,
                        help="Environment name")
    parser.add_argument("--action-type", type=str, default=None,
                        choices=['simple', 'rights', 'complex'],
                        help="Type of action space")
    parser.add_argument("--frame-stack", type=int, default=None,
                        help="Number of frames to stack")
    parser.add_argument("--frame-size", type=int, nargs=2, default=None,
                        metavar=("HEIGHT", "WIDTH"),
                        help="Frame size as height width")
    parser.add_argument("--frame-skip", type=int, default=None,
                        help="Number of frames to skip")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")
    parser.add_argument("--record-video", action="store_true",
                        help="Record gameplay video")
    parser.add_argument("--video-path", type=str, default=None,
                        help="Path to save video")
    parser.add_argument("--fps", type=int, default=None,
                        help="Frames per second for video")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Maximum steps for video recording")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")

    args = parser.parse_args()

    # Set random seeds if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    # Convert frame size to tuple if provided
    frame_size = tuple(args.frame_size) if args.frame_size else None

    if args.record_video:
        # Record gameplay video
        record_gameplay(
            model_path=args.model_path,
            env_name=args.env,
            action_type=args.action_type,
            frame_stack=args.frame_stack,
            frame_size=frame_size,
            frame_skip=args.frame_skip,
            output_path=args.video_path,
            max_steps=args.max_steps,
            fps=args.fps
        )
    else:
        # Standard evaluation
        mean_reward, mean_length = evaluate(
            model_path=args.model_path,
            env_name=args.env,
            action_type=args.action_type,
            frame_stack=args.frame_stack,
            frame_size=frame_size,
            frame_skip=args.frame_skip,
            n_episodes=args.episodes,
            render=not args.no_render,
            record_video=args.record_video,
            video_path=args.video_path,
            fps=args.fps
        )

        print(f"\nFinal Results:")
        print(f"Mean Reward: {mean_reward:.2f}")
        print(f"Mean Length: {mean_length:.2f}")


if __name__ == "__main__":
    main()
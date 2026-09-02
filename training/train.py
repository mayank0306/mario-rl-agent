"""
Main training loop for Mario RL Agent.
Currently implements random agent training - replace with actual DQN logic.
"""

import torch
import numpy as np
import random
import argparse
from typing import Tuple
import os
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from ..envs.mario_env import make_mario_env
from ..agents.dqn_agent import DQNAgent
from ..config import config


def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.tensorboard_dir, exist_ok=True)


def train(
    total_timesteps: int = None,
    env_name: str = None,
    action_type: str = None,
    frame_stack: int = None,
    frame_size: Tuple[int, int] = None,
    frame_skip: int = None,
    save_path: str = None,
    log_path: str = None,
    tb_path: str = None
):
    """
    Main training loop.

    Current implementation: Trains a random agent (placeholder)
    TODO: Replace with actual DQN training logic

    Args:
        total_timesteps: Total number of timesteps to train
        env_name: Name of the environment
        action_type: Type of action space
        frame_stack: Number of frames to stack
        frame_size: Size to resize frames to
        frame_skip: Number of frames to skip
        save_path: Path to save model checkpoints
        log_path: Path to save logs
        tb_path: Path for tensorboard logs
    """
    # Use config values if not provided
    total_timesteps = total_timesteps or config.total_timesteps
    env_name = env_name or config.env_name
    action_type = action_type or config.action_type
    frame_stack = frame_stack or config.frame_stack
    frame_size = frame_size or config.frame_size
    frame_skip = frame_skip or config.frame_skip
    save_path = save_path or os.path.join(config.checkpoint_dir, "mario_agent.pth")
    log_path = log_path or config.log_dir
    tb_path = tb_path or config.tensorboard_dir

    # Setup directories
    setup_directories()

    # Create environment
    env = make_mario_env(
        env_name=env_name,
        action_type=action_type,
        frame_stack=frame_stack,
        frame_size=frame_size,
        frame_skip=frame_skip
    )

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

    # Initialize tensorboard writer
    writer = SummaryWriter(tb_path)

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    best_mean_reward = -float('inf')

    # Training loop
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0

    for timestep in tqdm(range(1, total_timesteps + 1), desc="Training"):
        # Select action (currently random - TODO: implement epsilon-greedy)
        action = agent.select_action(state)

        # Take step in environment
        next_state, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1

        # Store experience
        agent.store_experience(state, action, reward, next_state, done or truncated)

        # Learn from experiences (currently placeholder - TODO: implement learning)
        loss = agent.learn()
        if loss is not None:
            writer.add_scalar('loss/train', loss, timestep)

        # Update epsilon
        agent.update_epsilon()

        # Update target network periodically
        if timestep % agent.target_update_freq == 0:
            agent.update_target_network()
            print(f"Timestep {timestep}: Updated target network")

        # Save checkpoint periodically
        if timestep % config.checkpoint_freq == 0:
            checkpoint_path = save_path.replace('.pth', f'_{timestep}.pth')
            agent.save_model(checkpoint_path)
            print(f"Timestep {timestep}: Saved checkpoint to {checkpoint_path}")

        # Log training progress
        if timestep % config.log_freq == 0:
            writer.add_scalar('epsilon', agent.epsilon, timestep)
            writer.add_scalar('steps_done', agent.steps_done, timestep)

        # Evaluate periodically
        if timestep % config.eval_freq == 0:
            mean_reward = evaluate_agent(agent, env, config.eval_episodes)
            writer.add_scalar('eval/mean_reward', mean_reward, timestep)
            writer.add_scalar('eval/mean_length', np.mean([ep_len for _, ep_len in []]), timestep)  # Placeholder

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                agent.save_model(save_path)
                print(f"Timestep {timestep}: New best model saved with reward {mean_reward:.2f}")

        # Reset if episode done
        if done or truncated:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Log episode metrics
            if len(episode_rewards) % 10 == 0:  # Log every 10 episodes
                mean_reward = np.mean(episode_rewards[-10:])
                mean_length = np.mean(episode_lengths[-10:])
                writer.add_scalar('train/episode_reward', episode_reward, len(episode_rewards))
                writer.add_scalar('train/episode_length', episode_length, len(episode_lengths))
                writer.add_scalar('train/mean_reward_10ep', mean_reward, len(episode_rewards))
                writer.add_scalar('train/mean_length_10ep', mean_length, len(episode_lengths))

            print(f"Timestep {timestep}: Episode finished - Reward: {episode_reward:.2f}, Length: {episode_length}")

            # Reset for next episode
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            agent.steps_done += 1
        else:
            state = next_state

    # Final evaluation and saving
    print("Training completed!")
    final_mean_reward = evaluate_agent(agent, env, config.eval_episodes)
    writer.add_scalar('final/mean_reward', final_mean_reward, total_timesteps)
    agent.save_model(save_path)
    print(f"Final model saved to {save_path}")
    print(f"Final mean reward: {final_mean_reward:.2f}")

    # Close environment and writer
    env.close()
    writer.close()


def evaluate_agent(agent: DQNAgent, env, n_episodes: int = 5) -> float:
    """
    Evaluate the agent's performance.

    Current implementation: Returns random reward (placeholder)
    TODO: Implement actual evaluation with learned policy

    Args:
        agent: The agent to evaluate
        env: Environment to evaluate in
        n_episodes: Number of episodes to evaluate for

    Returns:
        mean_reward: Average reward over evaluation episodes
    """
    # TODO: Implement actual evaluation:
    # 1. Set agent to evaluation mode (no exploration)
    # 2. Run n_episodes episodes
    # 3. Return mean reward

    # For now, return random reward as placeholder
    total_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            # TODO: Replace random action with agent's policy
            action = env.action_space.sample()  # Random action - placeholder
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)


def main():
    """Parse command line arguments and start training."""
    parser = argparse.ArgumentDescription("Train a DQN agent to play Super Mario Bros")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total timesteps for training")
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
    parser.add_argument("--save-path", type=str, default=None,
                        help="Path to save model")
    parser.add_argument("--log-path", type=str, default=None,
                        help="Path to save logs")
    parser.add_argument("--tb-path", type=str, default=None,
                        help="Path for tensorboard logs")
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

    # Start training
    train(
        total_timesteps=args.timesteps,
        env_name=args.env,
        action_type=args.action_type,
        frame_stack=args.frame_stack,
        frame_size=frame_size,
        frame_skip=args.frame_skip,
        save_path=args.save_path,
        log_path=args.log_path,
        tb_path=args.tb_path
    )


if __name__ == "__main__":
    main()
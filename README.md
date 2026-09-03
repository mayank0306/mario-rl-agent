<<<<<<< HEAD
# Mario RL Agent

A reinforcement learning agent (CNN + DQN/PPO) to play Super Mario Bros using gym-super-mario-bros and PyTorch.

## Folder Structure

```
mario-rl-agent/
├── README.md                 # project overview, setup instructions, how to run
├── requirements.txt          # torch, gymnasium, gym-super-mario-bros, nes-py, numpy, opencv-python, matplotlib, tensorboard
├── .gitignore                # python, venv, checkpoints, logs, __pycache__, .DS_Store
├── config.py                 # hyperparameters (learning rate, gamma, batch size, epsilon decay, etc.)
│
├── envs/
│   ├── __init__.py
│   └── mario_env.py          # wraps gym-super-mario-bros, handles preprocessing 
│                              # (grayscale, resize, frame stacking, frame skipping, 
│                              # action space simplification)
│
├── models/
│   ├── __init__.py
│   └── cnn.py                # CNN architecture (conv layers -> feature extractor -> Q-values or policy/value heads)
│
├── agents/
│   ├── __init__.py
│   ├── dqn_agent.py           # DQN agent: replay buffer, epsilon-greedy, target network, learn step
│   └── replay_buffer.py       # experience replay buffer implementation
│
├── training/
│   ├── __init__.py
│   ├── train.py               # main training loop, logging, checkpoint saving
│   └── evaluate.py            # load a trained model and watch/record it play
│
├── checkpoints/                # saved model weights (empty, gitkeep)
│   └── .gitkeep
│
├── logs/                        # tensorboard logs (empty, gitkeep)
│   └── .gitkeep
│
├── notebooks/
│   └── experiments.ipynb      # scratch space for testing environment, visualizing frames, debugging
│
└── scripts/
    ├── test_env.py             # quick script to verify the Mario environment works and render a few random steps
    └── record_gameplay.py      # record a video of the trained agent playing
```

## Installation

1. Clone the repository (or create the folder structure as above)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

## How to Run

### Test the Environment
Run the test script to verify the Mario environment works:
```bash
python scripts/test_env.py
```

### Start Training
Begin training the agent (currently with random actions - replace with DQN/PPO logic):
```bash
python training/train.py
```

### Evaluate a Trained Agent
After training, evaluate a saved model:
```bash
python training/evaluate.py --model_path checkpoints/mario_agent.pth
```

### Record Gameplay
Record a video of the trained agent playing:
```bash
python scripts/record_gameplay.py --model_path checkpoints/mario_agent.pth
```

## Configuration

Adjust hyperparameters in `config.py`:
- Learning rate, discount factor, batch size
- Epsilon decay rates for exploration
- Replay buffer size
- Frame stacking and preprocessing parameters

## Dependencies

See `requirements.txt` for the complete list:
- PyTorch
- Gymnasium
- gym-super-mario-bros
- nes-py
- NumPy
- OpenCV-Python
- Matplotlib
- TensorBoard

## Notes

- This project currently includes starter boilerplate with a random agent
- Replace the random actions in `dqn_agent.py` and `train.py` with actual DQN/PPO logic
- The CNN in `models/cnn.py` is designed for 84x84 grayscale stacked frames
- Checkpoints and TensorBoard logs are saved to their respective directories
=======
# mario-rl-agent
Reinforcement learning agent that learns to play Super Mario Bros through trial and error, using a CNN + DQN built from scratch in PyTorch.
>>>>>>> origin/main

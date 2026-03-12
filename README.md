# DRL-PID Vibration Control of a Spaceborne Flexible Beam

This project is the offical implementation of the paper (not published yet)
"Deep Reinforcement Learning-Based Gain-Scheduled PID Vibration Control of a Spaceborne Flexible Beam".

## Overview
This project trains a DDPG agent to output PID gains for a SISO beam vibration control task.
It includes:
- A state-space simulator with RK4 integration
- Disturbance generation (jitter, maneuver, impact, thermal, mixed)
- DDPG training with expert data, behavior cloning, and robust rollback
- Baseline and PSO utilities for comparison

## Project Structure
- `config/`
  - `dynamics_model_config.yaml`: system parameters
  - `config.py`: training and environment config
  - `config_loader.py`: YAML and MAT loader
- `src/`
  - `agents/`: DDPG actor/critic and replay buffers
  - `environments/`: PID control RL environment
  - `solvers/`: state-space solvers and baselines
  - `utils/`: noise generation and projector utilities
- `scripts/`
  - `train_ddgp_pid.py`: main DDPG training script
- `tests/`
  - `state_space_pso_pid.py`: PSO-based PID search
  - `test_ddpg_pid.py`: baseline comparison and plotting
- `models/`: saved checkpoints
- `results/`: output figures and logs

## Requirements
Install Python dependencies:
```bash
pip install -r requirements.txt
```

PyTorch is required but not pinned in `requirements.txt`. Install it separately to match your CPU/CUDA setup.

## Configuration
- System parameters live in `config/dynamics_model_config.yaml`.
- Thermal disturbance data uses `config/BendingMomentResult.mat` if available.

## Usage

### Train DDPG
```bash
python scripts/train_ddgp_pid.py
```

### PSO PID Search (parallel)
```bash
python tests/state_space_pso_pid.py
```

### Baseline Comparison and Plots
```bash
python tests/test_ddpg_pid.py
```

## Notes
- `results/` is used for plot outputs.
- `models/` stores intermediate and final checkpoints.

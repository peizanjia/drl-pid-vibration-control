import torch
import numpy as np
from config import config_loader


class Config:
    """Configuration parameters for DDPG-based adaptive PID control."""

    # Environment parameters
    DT = 0.01  # Sampling time
    EPISODE_LENGTH = 10000  # Steps per episode
    EPISODES = 300  # Number of training episodes

    # Dynamics system parameters
    SYSTEM_CONFIG = config_loader.load_config()
    THERMAL_MOMENT = config_loader.load_mat()

    # DDPG parameters
    STATE_DIM = 5  # [u, e, ei, ed, t_normalized]
    ACTION_DIM = 3  # [kp, ki, kd]

    # PID parameter ranges (before normalization)
    KP_RANGE = [0.0, 200.0]
    KI_RANGE = [0.0, 30.0]
    KD_RANGE = [20.0, 30.0]

    U_LIMIT = 300.0

    # Network parameters
    HIDDEN_DIM = 128
    ACTOR_LR = 1e-5
    CRITIC_LR = 1e-4

    # Training parameters
    GAMMA = 0.9999
    TAU = 0.001
    BUFFER_SIZE = 150000
    BATCH_SIZE = 1024

    # Reward weights
    OUTPUT_WEIGHT = 1  # Output Y weight
    PARAM_PENALTY = 10  # Penalty for parameter changes

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        # Precompute normalization parameters
        self.action_mean = np.array([
            (self.KP_RANGE[0] + self.KP_RANGE[1]) / 2,
            (self.KI_RANGE[0] + self.KI_RANGE[1]) / 2,
            (self.KD_RANGE[0] + self.KD_RANGE[1]) / 2
        ])

        self.action_std = np.array([
            (self.KP_RANGE[1] - self.KP_RANGE[0]) / 2,
            (self.KI_RANGE[1] - self.KI_RANGE[0]) / 2,
            (self.KD_RANGE[1] - self.KD_RANGE[0]) / 2
        ])

    def normalize_action(self, action):
        """Normalize action to [-1, 1]."""
        normalized_action = np.zeros(3)
        for i in range(3):
            if self.action_std[i] != 0:
                normalized_action[i] = (action[i] - self.action_mean[i]) / self.action_std[i]
        return normalized_action

    def denormalize_action(self, normalized_action):
        """Denormalize action back to physical PID gains."""
        return normalized_action * self.action_std + self.action_mean

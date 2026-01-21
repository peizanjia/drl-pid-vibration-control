import torch
import numpy as np
from config import config_loader


class Config:
    """DDPG自适应PID控制的配置参数"""

    # 环境参数
    DT = 0.01  # 采样时间
    EPISODE_LENGTH = 10000  # 每个episode的长度
    EPISODES = 300  # 训练episode数量

    # 动力学系统参数
    SYSTEM_CONFIG = config_loader.load_config()
    THERMAL_MOMENT = config_loader.load_mat()

    # DDPG参数
    STATE_DIM = 5  # [Y, e, ei, ed, t_normalized]
    ACTION_DIM = 3  # [kp, ki, kd]

    # PID参数范围（归一化前）
    KP_RANGE = [0.0, 200.0]
    KI_RANGE = [0.0, 10.0]
    KD_RANGE = [10.0, 25.0]

    # 网络参数
    HIDDEN_DIM = 128
    ACTOR_LR = 1e-4
    CRITIC_LR = 1e-3

    # 训练参数
    GAMMA = 0.99
    TAU = 0.001
    BUFFER_SIZE = 100000
    BATCH_SIZE = 64
    WARMUP_STEPS = 7500

    # 奖励函数权重
    OUTPUT_WEIGHT = 1  # 输出Y的权重
    CONTROL_WEIGHT = 0  # 控制力的权重
    PARAM_PENALTY = 10  # 参数变化惩罚

    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        # 计算归一化参数
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
        """将动作归一化到[-1, 1]范围"""
        normalized_action = np.zeros(3)
        for i in range(3):
            if self.action_std[i] != 0:
                normalized_action[i] = (action[i] - self.action_mean[i]) / self.action_std[i]
        return normalized_action

    def denormalize_action(self, normalized_action):
        """将归一化动作转换回实际PID参数"""
        return normalized_action * self.action_std + self.action_mean
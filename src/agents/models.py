import torch
import torch.nn as nn


def init_weights(m):
    """方案二：正交初始化 - 保持层与层之间特征分布的方差一致"""
    if isinstance(m, nn.Linear):
        # 使用正交初始化，gain 根据激活函数调整，ReLU 系列通常用 sqrt(2)
        nn.init.orthogonal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0.0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()

        # 方案一：引入 LayerNorm 和 LeakyReLU
        # LayerNorm 不需要改变输入维度，它直接在 hidden_dim 上做标准化
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        # 执行初始化
        self.apply(init_weights)

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()

        # Critic 同样需要加固，防止 Q 值爆炸
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        self.action_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)  # 输出层通常不需要激活函数，也不需要标准化
        )
        self.apply(init_weights)

    def forward(self, state, action):
        state_out = self.state_net(state)
        action_out = self.action_net(action)
        combined = torch.cat([state_out, action_out], dim=1)
        return self.combined_net(combined)
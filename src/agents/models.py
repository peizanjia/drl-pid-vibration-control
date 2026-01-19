import torch
import torch.nn as nn



class Actor(nn.Module):
    """Actor网络 - 输出PID参数"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出在[-1, 1]范围内
        )

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    """Critic网络 - Q值函数"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()

        # 状态路径
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 动作路径
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU()
        )

        # 合并路径
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        state_out = self.state_net(state)
        action_out = self.action_net(action)
        combined = torch.cat([state_out, action_out], dim=1)
        return self.combined_net(combined)
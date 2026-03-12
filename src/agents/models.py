import torch
import torch.nn as nn


def init_weights(m):
    """Orthogonal init to keep variance consistent across layers."""
    if isinstance(m, nn.Linear):
        # Orthogonal init with ReLU gain (works well with LeakyReLU too)
        nn.init.orthogonal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0.0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()

        # Option A: LayerNorm + LeakyReLU stack
        # LayerNorm works directly on hidden_dim without changing input size
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
        # Initialize weights
        self.apply(init_weights)

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()

        # Critic should be stabilized to avoid Q-value explosion
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
            nn.Linear(hidden_dim, 1)  # No activation needed for output
        )
        self.apply(init_weights)

    def forward(self, state, action):
        state_out = self.state_net(state)
        action_out = self.action_net(action)
        combined = torch.cat([state_out, action_out], dim=1)
        return self.combined_net(combined)

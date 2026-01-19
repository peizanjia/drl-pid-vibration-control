import torch
import torch.optim as optim
import numpy as np
import random
from collections import deque
from .models import Actor
from .models import Critic
import torch.nn.functional as F


class OUNoise:
    """Ornstein-Uhlenbeck过程，用于探索"""

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dt=0.01):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = self.mu.copy()
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    """DDPG智能体"""

    def __init__(self, config):
        self.config = config

        # 网络
        self.actor = Actor(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(config.DEVICE)
        self.actor_target = Actor(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(config.DEVICE)
        self.critic = Critic(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(config.DEVICE)
        self.critic_target = Critic(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(config.DEVICE)

        # 硬拷贝参数到目标网络
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

        # 经验回放
        self.memory = ReplayBuffer(config.BUFFER_SIZE)

        # 探索噪声
        self.noise = OUNoise(config.ACTION_DIM,
                             theta=0.15,
                             sigma=0.2,
                             dt=config.DT)

    def hard_update(self, target, source):
        """硬更新目标网络参数"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        """软更新目标网络参数"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def select_action(self, state, add_noise=True):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.DEVICE)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        self.actor.train()

        if add_noise:
            action += self.noise.sample()
            action = np.clip(action, -1.0, 1.0)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """存储转移经验"""
        self.memory.add(state, action, reward, next_state, done)

    def update_networks(self):
        """更新网络参数"""
        BATCH_SIZE = min(len(self.memory), self.config.BATCH_SIZE)

        # 从回放缓冲区采样
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states = torch.FloatTensor(states).to(self.config.DEVICE)
        actions = torch.FloatTensor(actions).to(self.config.DEVICE)
        rewards = torch.FloatTensor(rewards).to(self.config.DEVICE)
        next_states = torch.FloatTensor(next_states).to(self.config.DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.config.DEVICE)

        # 更新Critic网络
        next_actions = self.actor_target(next_states)
        next_Q_values = self.critic_target(next_states, next_actions.detach())
        target_Q_values = rewards + (1 - dones) * self.config.GAMMA * next_Q_values

        current_Q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q_values, target_Q_values.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 更新Actor网络
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # 软更新目标网络
        self.soft_update(self.actor_target, self.actor, self.config.TAU)
        self.soft_update(self.critic_target, self.critic, self.config.TAU)

        return critic_loss.item(), actor_loss.item()

    def save_models(self, filepath):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)

    def load_models(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.config.DEVICE)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
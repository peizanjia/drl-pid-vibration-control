import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from .models import Actor, Critic


class OUNoise:
    """向量化的 Ornstein-Uhlenbeck 噪声 process"""

    def __init__(self, action_dim, num_envs=1, mu=0.0, theta=0.15, sigma=0.2, dt=0.01):
        self.action_dim = action_dim
        self.num_envs = num_envs
        self.mu = mu * np.ones((num_envs, action_dim))
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = self.mu.copy()
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        """返回 (num_envs, action_dim) 形状的噪声"""
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
        # 兼容单条存储
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, config, num_envs=1):  # 增加 num_envs 参数
        self.config = config
        self.num_envs = num_envs

        self.actor = Actor(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(config.DEVICE)
        self.actor_target = Actor(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(config.DEVICE)
        self.critic = Critic(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(config.DEVICE)
        self.critic_target = Critic(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(config.DEVICE)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

        self.memory = ReplayBuffer(config.BUFFER_SIZE)

        # 向量化噪声
        self.noise = OUNoise(config.ACTION_DIM,
                             num_envs=num_envs,  # 传入环境数量
                             theta=0.3,
                             sigma=0.02,
                             dt=config.DT)

        # 延迟更新计数器
        self.update_cnt = 0
        self.policy_freq = 2  # 甚至 Critic 更新 2 次，Actor 更新 1 次

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def select_action(self, state, add_noise=True):
        """
        支持 Batch 处理
        state: (state_dim,) or (num_envs, state_dim)
        Returns: (action_dim,) or (num_envs, action_dim)
        """
        # 统一转为 tensor
        state = np.array(state)
        if state.ndim == 1:
            state = state[None, :]  # (1, state_dim)

        state_tensor = torch.FloatTensor(state).to(self.config.DEVICE)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy()
        self.actor.train()

        if add_noise:
            # 噪声也是 (num_envs, action_dim)
            noise_sample = self.noise.sample()
            # 如果当前是单条推理（比如测试时），只取噪声的第一行
            if action.shape[0] == 1 and noise_sample.shape[0] != 1:
                action += noise_sample[0]
            else:
                action += noise_sample

            action = np.clip(action, -1.0, 1.0)

        # 如果输入是单个状态，返回单个动作；否则返回 Batch
        if action.shape[0] == 1:
            return action.flatten()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def update_networks(self, update_actor=True, critic_iters=1):
        """
        Args:
            update_actor: 是否更新 Actor (用于延迟更新)
            critic_iters: Critic 更新次数
        """
        if len(self.memory) < self.config.BATCH_SIZE:
            return None, None

        critic_loss_val = 0

        # --- 循环更新 Critic 多次 ---
        for i in range(critic_iters):
            states, actions, rewards, next_states, dones = self.memory.sample(self.config.BATCH_SIZE)

            states = torch.FloatTensor(states).to(self.config.DEVICE)
            actions = torch.FloatTensor(actions).to(self.config.DEVICE)
            rewards = torch.FloatTensor(rewards).view(-1, 1).to(self.config.DEVICE)
            next_states = torch.FloatTensor(next_states).to(self.config.DEVICE)
            dones = torch.FloatTensor(dones).view(-1, 1).to(self.config.DEVICE)

            # 更新 Critic
            with torch.no_grad():
                next_actions = self.actor_target(next_states)
                # 可以加入 Target Smoothing Noise
                noise = torch.randn_like(next_actions) * 0.2
                noise = noise.clamp(-0.5, 0.5)
                next_actions = (next_actions + noise).clamp(-1.0, 1.0)

                next_Q_values = self.critic_target(next_states, next_actions)
                target_Q_values = rewards + (1 - dones) * self.config.GAMMA * next_Q_values

            current_Q_values = self.critic(states, actions)
            critic_loss = F.mse_loss(current_Q_values, target_Q_values)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # 梯度裁剪很重要
            self.critic_optimizer.step()

            critic_loss_val += critic_loss.item()

        # --- 延迟更新 Actor (只做一次) ---
        actor_loss_val = None
        if update_actor:
            # 重新采样一次或者沿用最后一次的数据都可以，通常重新采样
            # 为了省事，这里沿用最后一次的 states
            actor_actions = self.actor(states)
            actor_loss = -self.critic(states, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # 软更新
            self.soft_update(self.actor_target, self.actor, self.config.TAU)
            self.soft_update(self.critic_target, self.critic, self.config.TAU)

            actor_loss_val = actor_loss.item()

        return critic_loss_val / critic_iters, actor_loss_val

    def save_models(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
        }, filepath)

    def load_models(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.config.DEVICE)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_opt'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_opt'])
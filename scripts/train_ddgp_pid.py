import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path

# 项目内部导入
from src.environments.environment_2 import PIDControlEnvironment
from src.agents.ddpg_agent import DDPGAgent
from src.solvers.state_space_2 import StateSpace
from config.config import Config
from config.config_loader import load_mat
from src.utils.utils import BeamDisturbanceProjector, create_noise_data

# 设置路径
ROOT_DIR = Path(__file__).resolve().parent.parent  # 假设 train_parallel 在 scripts 目录下
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def worker(remote, parent_remote, config, noise_type, mt_data, projector_coeffs):
    """
    Worker 进程循环：接收动作 -> 环境步进 -> 发送状态
    """
    parent_remote.close()

    # 局部实例化环境
    tn = config.EPISODE_LENGTH

    # 辅助函数：重新生成环境（用于 Reset）
    def make_env():
        # 调用 utils.py 中修改后的 create_noise_data
        noise_dict = create_noise_data(
            tn=tn,
            dt=config.DT,
            option=noise_type,
            system_config=config.SYSTEM_CONFIG,
            mt_data=mt_data,
            projector_data=projector_coeffs  # 传入静态系数
        )
        ss = StateSpace(config.SYSTEM_CONFIG, noise_dict, dt=config.DT, tn=tn)
        env = PIDControlEnvironment(config)
        env.set_state_space(ss)
        return env

    env = make_env()

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                action = data
                next_state, reward, done, info = env.step(action)
                if done:
                    # 如果 Done，自动 Reset (Reinforcement Learning 常见做法)
                    # 重新生成噪声数据以保证下一回合不同
                    env = make_env()
                    next_state = env.reset()
                remote.send((next_state, reward, done, info))

            elif cmd == 'reset':
                env = make_env()
                state = env.reset()
                remote.send(state)

            elif cmd == 'get_history':
                # 返回用于绘图的数据
                remote.send((env.state_space.Y, env.state_space.Reference))

            elif cmd == 'close':
                remote.close()
                break
    except Exception as e:
        print(f"Worker Error ({noise_type}): {e}")


class ParallelEnv:
    def __init__(self, config, num_envs=15):
        self.config = config

        # 1. 准备数据 (主进程加载一次)
        print("Loading thermal data...")
        try:
            mt_data = load_mat()
        except:
            print("Warning: Could not load thermal data. Thermal mode may fail.")
            mt_data = None

        # 2. 准备 Projector 系数 (主进程计算一次)
        print("Pre-calculating projector coefficients...")
        proj = BeamDisturbanceProjector()
        projector_coeffs = proj.get_static_coeffs()

        # 3. 分配模式 (3*5 = 15 envs)
        base_modes = ['jitter', 'maneuver', 'impact', 'mixed', 'thermal']
        self.mode_list = (base_modes * 3)[:num_envs]  # 确保数量匹配
        self.num_envs = len(self.mode_list)

        print(f"Initializing {self.num_envs} environments with modes: {self.mode_list}")

        # 4. 启动进程
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.ps = []
        for i, (work_remote, remote, mode) in enumerate(zip(self.work_remotes, self.remotes, self.mode_list)):
            p = mp.Process(target=worker,
                           args=(work_remote, remote, config, mode, mt_data, projector_coeffs))
            p.daemon = True
            p.start()
            self.ps.append(p)
            work_remote.close()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_history(self, env_idx):
        self.remotes[env_idx].send(('get_history', None))
        return self.remotes[env_idx].recv()

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()


def train():
    config = Config()

    # 初始化并行环境 (15个核心)
    envs = ParallelEnv(config, num_envs=15)

    # 初始化 Agent (注意传入 num_envs 用于噪声向量化)
    agent = DDPGAgent(config, num_envs=envs.num_envs)

    print("Start Training Parallel DDPG...")
    total_steps = 0
    start_time = time.time()

    # 记录器
    history_rewards = []

    for episode in range(config.EPISODES):
        states = envs.reset()
        agent.noise.reset()

        ep_rewards = np.zeros(envs.num_envs)

        for step in range(config.EPISODE_LENGTH):
            # 1. 动作选择 (Batch)
            if total_steps < config.WARMUP_STEPS:
                actions = np.random.uniform(-1, 1, (envs.num_envs, config.ACTION_DIM))
            else:
                actions = agent.select_action(states)  # Returns (15, action_dim)

            # 2. 环境交互
            next_states, rewards, dones, infos = envs.step(actions)

            # 3. 存储经验 (Loop storage)
            # 因为 ReplayBuffer 是线性的，这里我们需要拆开 batch 存进去
            # 或者修改 ReplayBuffer 支持 batch add。这里简单循环即可，CPU 很快。
            for i in range(envs.num_envs):
                agent.store_transition(states[i], actions[i], rewards[i], next_states[i], dones[i])
                r=rewards[i]
                ep_rewards[i] += rewards[i].item()

            # 4. 更新网络
            # 你的要求：每步都更新 Critic 吗？
            # 建议：如果采集了 15 条数据，可以更新 1 次网络，或者更多。
            # 这里设置：每一步并行采集完，Update 一次 (Batch Size=128)
            if len(agent.memory) > config.BATCH_SIZE:
                agent.update_networks()  # 内部已包含 Actor Delay 逻辑

            states = next_states
            total_steps += envs.num_envs  # 这一步实际上走了 15 个 step

        # Episode 结束
        avg_ep_reward = np.mean(ep_rewards)
        history_rewards.append(avg_ep_reward)

        if episode % 10 == 0:
            print(f"Episode {episode} | Avg Reward: {avg_ep_reward:.2f} | Steps: {total_steps}")
            agent.save_models(MODELS_DIR / f'ddpg_ep_{episode}.pth')

            # 绘图检查: 随机抽一个 impact 环境 (mode_list index 2, 7, 12 是 impact)
            try:
                # 找第一个 impact 环境的索引
                impact_idx = envs.mode_list.index('impact')
                Y, Ref = envs.get_history(impact_idx)
                plt.figure(figsize=(10, 5))
                plt.plot(Y, label='Response')
                plt.title(f'Impact Response (Ep {episode})')
                plt.legend()
                plt.savefig(RESULTS_DIR / f'resp_ep_{episode}.png')
                plt.close()
            except:
                pass

    print(f"Training Finished. Time: {(time.time() - start_time) / 60:.1f} min")
    agent.save_models(MODELS_DIR / 'ddpg_final.pth')
    envs.close()

    # 绘制最终曲线
    plt.plot(history_rewards)
    plt.savefig(RESULTS_DIR / 'training_curve.png')


if __name__ == '__main__':
    train()
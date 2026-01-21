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

    # 1. 初始化
    envs = ParallelEnv(config, num_envs=15)
    agent = DDPGAgent(config, num_envs=envs.num_envs)  # 你的 DDPG 代码保持不变

    print("Start Training with Elite Strategy & NaN-Protection...")
    total_steps = 0
    start_time = time.time()

    # 记录器
    history_rewards = []

    # --- 核心数据结构：轨迹缓冲区 ---
    # 结构: lists of (state, action, reward, next_state, done)
    # 每个环境拥有一个独立的缓存列表
    trajectory_buffers = [[] for _ in range(envs.num_envs)]

    # 状态追踪器
    # 记录当前回合每个环境的累计奖励，用于判断是否为"精英"
    current_ep_rewards = np.zeros(envs.num_envs)
    # 记录每个环境是否"存活" (没有遇到 NaN)
    env_active_mask = np.ones(envs.num_envs, dtype=bool)

    for episode in range(config.EPISODES):
        # Reset 之后，得到的 states 是干净的
        states = envs.reset()
        agent.noise.reset()

        # 重置临时缓冲区和追踪器
        for i in range(envs.num_envs):
            trajectory_buffers[i] = []
        current_ep_rewards.fill(0)
        env_active_mask.fill(True)

        # 统计本回合有多少个环境成功存活并被学习
        successful_envs_count = 0

        for step in range(config.EPISODE_LENGTH):
            # -------------------------------------------------
            # 1. 安全的动作选择 (NaN 隔离防火墙)
            # -------------------------------------------------
            # 检查 states 是否含有 NaN
            clean_states = states.copy()
            nan_indices = np.isnan(states).any(axis=1)

            # 如果发现 NaN，将其替换为 0，防止网络前向传播时崩溃
            # 注意：这些环境已经被标记为 inactive，它们的输出 action 我们稍后会无视
            if np.any(nan_indices):
                clean_states[nan_indices] = 0.0
                # 这里不更新 mask，因为 mask 由这一步的交互结果决定，
                # 但如果是上一轮传下来的 NaN，已经在上一轮被 mask 掉了

            if total_steps < config.WARMUP_STEPS:
                actions = np.random.uniform(-1, 1, (envs.num_envs, config.ACTION_DIM))
            else:
                # 使用清洗过的 state 进行推理
                actions = agent.select_action(clean_states)

                # -------------------------------------------------
            # 2. 环境交互
            # -------------------------------------------------
            next_states, rewards, dones, infos = envs.step(actions)

            # -------------------------------------------------
            # 3. 精英策略与 NaN 过滤 (核心逻辑)
            # -------------------------------------------------
            for i in range(envs.num_envs):
                # 如果这个环境在之前步骤已经挂了，直接跳过
                if not env_active_mask[i]:
                    continue

                # 检查当前步是否产生 NaN (爆炸检测)
                # 只要 s', r 中有任何一个是 NaN，立刻判死刑
                if np.isnan(next_states[i]).any() or np.isnan(rewards[i]) or np.isinf(rewards[i]):
                    env_active_mask[i] = False
                    trajectory_buffers[i].clear()  # 【整段放弃】：清空之前存的所有步数
                    # 可选：如果你希望 Agent 知道这里很危险，可以存入最后一步并给巨额惩罚
                    # 但"精英策略"通常选择直接无视垃圾数据
                    continue

                # 如果存活，暂存入临时 Buffer
                # 注意：这里存的是 Python float，用 .item() 避免 numpy 警告
                r_val = rewards[i].item()
                trajectory_buffers[i].append(
                    (states[i], actions[i], r_val, next_states[i], dones[i])
                )
                current_ep_rewards[i] += r_val

            # -------------------------------------------------
            # 4. 网络更新 (解耦)
            # -------------------------------------------------
            # 只要 Memory 里有数据就可以更新，不依赖于当前步是否有人爆炸
            if len(agent.memory) > config.BATCH_SIZE:
                agent.update_networks()

            states = next_states
            total_steps += envs.num_envs

        # === Episode 结束 ===

        # 5. 将“幸存者”的经验转正 (Flush Buffer)
        for i in range(envs.num_envs):
            if env_active_mask[i]:  # 只有全程存活的环境才有资格进入记忆库
                successful_envs_count += 1

                # 【精英策略扩展】：你可以在这里加更严格的判断
                # 例如：if current_ep_rewards[i] > -5000: 才存入
                for transition in trajectory_buffers[i]:
                    agent.store_transition(*transition)

        # 6. 计算统计数据 (只统计幸存者)
        if successful_envs_count > 0:
            # 既然被放弃的环境 buffer 都清空了，rewards 也不能算它们
            # 这里我们取 active mask 对应的 reward 均值
            avg_ep_reward = np.mean(current_ep_rewards[env_active_mask])
        else:
            avg_ep_reward = -99999.0  # 全军覆没

        history_rewards.append(avg_ep_reward)

        # 7. 打印与保存
        print(
            f"Ep {episode:3d} | Valid: {successful_envs_count:2d}/15 | Avg Reward: {avg_ep_reward:8.2f} | Steps: {total_steps}")

        if episode % 10 == 0:
            agent.save_models(MODELS_DIR / f'ddpg_ep_{episode}.pth')
            # ... 绘图代码保持不变 ...
            try:
                # 绘图逻辑同前
                impact_idx = envs.mode_list.index('impact')
                Y, Ref = envs.get_history(impact_idx)
                plt.figure(figsize=(10, 5))
                plt.plot(Y, label='Response')
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
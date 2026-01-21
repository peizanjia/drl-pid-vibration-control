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
    def make_env(seed=None):
        if seed is not None:
            np.random.seed(seed)
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
                # --- 修改点：这里去掉了自动 Reset ---
                # 即使 done=True，也只返回状态，保持 env 对象不变
                # 以便主进程调用 get_history
                remote.send((next_state, reward, done, info))

            elif cmd == 'reset':
                # 主进程显式要求 Reset
                seed = data  # 接收主进程传来的 seed
                env = make_env(seed)
                state = env.reset()
                remote.send(state)

            elif cmd == 'get_history':
                # 此时 env 还是旧的，数据还在
                # 返回：响应Y, 参考Ref, PID参数数组(Kp, Ki, Kd)
                hist_kp = env.state_space.kp
                hist_ki = env.state_space.ki
                hist_kd = env.state_space.kd
                remote.send((env.state_space.Y, env.state_space.Reference,
                             hist_kp, hist_ki, hist_kd))

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

        # reset 需要支持传入种子列表，或者保持随机
    def reset(self, seeds=None):
        if seeds is None:
            seeds = [None] * self.num_envs

        for remote, seed in zip(self.remotes, seeds):
            remote.send(('reset', seed))
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
    start_time = time.time()
    history_rewards = []
    envs = ParallelEnv(config, num_envs=15)
    # 注意：num_envs 只影响噪声初始化，select_action 改成单次调用后，这里也可以传 1，但传 15 也没坏处
    agent = DDPGAgent(config, num_envs=1)

    # 核心缓冲与追踪
    trajectory_buffers = [[] for _ in range(envs.num_envs)]
    current_ep_rewards = np.zeros(envs.num_envs)
    env_active_mask = np.ones(envs.num_envs, dtype=bool)

    # 记录每个环境每回合的失稳原因
    failure_reasons = [""] * envs.num_envs

    print("Start Training (Decoupled & Robust Mode)...")

    for episode in range(config.EPISODES):
        # 1. 重置
        # 可以在这里通过传入 seeds 来固定扰动时间，例如 seeds=[episode]*15
        states = envs.reset()
        agent.noise.reset()

        for i in range(envs.num_envs): trajectory_buffers[i] = []
        current_ep_rewards.fill(0)
        env_active_mask.fill(True)
        failure_reasons = [""] * envs.num_envs

        total_steps_in_episode = 0

        for step in range(config.EPISODE_LENGTH):

            # --- [解耦] 串行推理动作 ---
            # 即使很慢，但绝对安全
            actions = np.zeros((envs.num_envs, config.ACTION_DIM))

            for i in range(envs.num_envs):
                if env_active_mask[i]:
                    # 检查输入状态是否正常
                    if np.isnan(states[i]).any():
                        env_active_mask[i] = False
                        failure_reasons[i] = "State NaN"
                        continue

                    # 推理
                    action = agent.select_action(states[i], add_noise=True)

                    # 检查输出动作是否正常
                    if np.isnan(action).any():
                        env_active_mask[i] = False
                        failure_reasons[i] = "Action NaN"
                        actions[i] = np.zeros(config.ACTION_DIM)  # 默认输出
                    else:
                        actions[i] = action
                else:
                    actions[i] = np.zeros(config.ACTION_DIM)  # 死亡环境输出0

            # --- [并行] 环境步进 ---
            next_states, rewards, dones, infos = envs.step(actions)

            # --- [审计] 数据检查与缓存 ---
            for i in range(envs.num_envs):
                if not env_active_mask[i]: continue

                r_val = rewards[i].item()

                # 检查 NaN 爆炸
                if np.isnan(next_states[i]).any() or np.isnan(r_val) or np.isinf(r_val):
                    env_active_mask[i] = False
                    failure_reasons[i] = "Reward/NextState NaN"

                    # 【关键策略】：与其整段放弃，不如记录"导致死亡的一步"
                    # 存入一个带有巨大惩罚的 transition，告诉 Critic 这里是悬崖
                    # next_state 用全0填充，防止污染
                    clean_next_state = np.zeros_like(states[i])
                    death_penalty = -100.0  # 显式惩罚

                    trajectory_buffers[i].append(
                        (states[i], actions[i], death_penalty, clean_next_state, True)  # Done=True
                    )
                    # 之后的轨迹不再记录
                    continue

                # 正常存入
                trajectory_buffers[i].append(
                    (states[i], actions[i], r_val, next_states[i], dones[i])
                )
                current_ep_rewards[i] += r_val

            # --- [更新] 每次 update 3次 Critic ---
            if len(agent.memory) > config.BATCH_SIZE:
                agent.update_networks(update_actor=True, critic_iters=3)

            states = next_states
            total_steps_in_episode += 1

        # === Episode 结束 ===

        # 1. 存入经验 (精英筛选可加在这里)
        # 简单策略：只要没因 NaN 挂掉的轨迹都存入；或者挂掉的也存入了(含死亡惩罚)
        valid_count = 0
        for i in range(envs.num_envs):
            # 如果仅仅是存入 Buffer，可以将 trajectory_buffers[i] 全部压入
            # 即使是中途挂掉的，我们上面也已经处理了最后一步
            if len(trajectory_buffers[i]) > 0:
                for t in trajectory_buffers[i]:
                    agent.store_transition(*t)

            if env_active_mask[i]:
                valid_count += 1

        # 2. 统计失效模式
        # 统计有哪些模式挂了
        failed_modes = []
        for i in range(envs.num_envs):
            if not env_active_mask[i]:
                mode_name = envs.mode_list[i]
                failed_modes.append(f"{mode_name}({failure_reasons[i]})")

        # 计算有效组的平均分
        if valid_count > 0:
            avg_reward = np.mean([current_ep_rewards[i] for i in range(envs.num_envs) if env_active_mask[i]])
        else:
            avg_reward = np.nan

        history_rewards.append(avg_reward)

        # 3. 打印信息
        print(f"Ep {episode:3d} | Valid: {valid_count:2d}/15 | AvgR: {avg_reward:7.1f} | Fail: {failed_modes[:3]}...")

        # 4. 绘图与保存 (每10回合)
        if episode % 10 == 0:
            agent.save_models(MODELS_DIR / f'ddpg_ep_{episode}.pth')

            # 定义需要画图的模式
            target_modes = ['impact', 'mixed']

            for target_mode in target_modes:
                try:
                    # 找到该模式对应的第一个环境索引
                    if target_mode in envs.mode_list:
                        idx = envs.mode_list.index(target_mode)

                        # 获取完整历史 (包含 PID)
                        Y, Ref, Kp, Ki, Kd = envs.get_history(idx)

                        # 绘图
                        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

                        # 子图1: 响应
                        axes[0].plot(Ref, 'k--', label='Ref', alpha=0.5)
                        axes[0].plot(Y, 'b', label='Response')
                        axes[0].set_title(f'Ep {episode} - {target_mode} Response (R={current_ep_rewards[idx]:.1f})')
                        axes[0].legend()
                        axes[0].grid(True)

                        # 子图2: PID参数
                        axes[1].plot(Kp, label='Kp')
                        axes[1].plot(Ki, label='Ki')
                        axes[1].plot(Kd, label='Kd')
                        axes[1].set_title('PID Gains Evolution')
                        axes[1].legend()
                        axes[1].grid(True)

                        plt.tight_layout()
                        plt.savefig(RESULTS_DIR / f'resp_{target_mode}_ep_{episode}.png')
                        plt.close()
                except Exception as e:
                    print(f"Plotting failed for {target_mode}: {e}")

    # === 训练结束阶段 ===
    print(f"Training Finished. Total Time: {(time.time() - start_time) / 60:.1f} min")

    # 1. 保存最终模型 (Final Model)
    final_model_path = MODELS_DIR / 'ddpg_final_robust.pth'
    agent.save_models(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # 2. 绘制并保存最终训练曲线
    plt.figure(figsize=(12, 6))

    # 原始奖励数据
    rewards_np = np.array(history_rewards)
    # 绘图：Matplotlib 会自动断开 NaN 点
    plt.plot(rewards_np, color='royalblue', linewidth=1.5, label='Average Reward')

    # 为了让缺口更明显，可以在 NaN 处标记特殊颜色（可选）
    nan_indices = np.where(np.isnan(rewards_np))[0]
    if len(nan_indices) > 0:
        # 在 X 轴底部画红点表示“系统爆炸”
        plt.scatter(nan_indices, [np.nanmin(rewards_np)] * len(nan_indices),
                    color='red', marker='x', s=20, label='System Failure (NaN)')

    plt.title('DDPG Training Progress - Real-time Control Stability')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward of Valid Envs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存结果
    plt.savefig(RESULTS_DIR / 'final_training_raw_curve.png')
    plt.show()

    # 3. 统计最终的“生存率”分布
    # 这里的分析能帮你判断：哪些扰动模式目前仍然是 Agent 的“盲区”
    print("\n--- Final Training Summary ---")
    print(f"Total Episodes: {config.EPISODES}")
    print(f"Final Reward: {rewards_np[-1] if 'rewards_np' in locals() else 'N/A':.2f}")

    envs.close()


if __name__ == '__main__':
    train()
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
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def worker(remote, parent_remote, config, noise_type, mt_data, projector_coeffs):
    parent_remote.close()
    tn = config.EPISODE_LENGTH

    def make_env(seed=None):
        if seed is not None: np.random.seed(seed)
        noise_dict = create_noise_data(
            tn=tn, dt=config.DT, option=noise_type,
            system_config=config.SYSTEM_CONFIG, mt_data=mt_data,
            projector_data=projector_coeffs
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
                action = data  # 接收的是归一化动作
                next_state, reward, done, info = env.step(action)
                # 将物理响应 Y 传回用于发散判定
                # 假设 env.state_space.Y 记录了当前所有步的响应
                current_y = env.state_space.Y[env.state_space.current_step - 1]
                info['abs_y'] = abs(current_y)
                remote.send((next_state, reward, done, info))

            elif cmd == 'reset':
                env = make_env(data)
                state = env.reset()
                remote.send(state)

            elif cmd == 'get_history':
                remote.send((env.state_space.Y, env.state_space.Reference,
                             env.state_space.kp, env.state_space.ki, env.state_space.kd))

            elif cmd == 'close':
                remote.close()
                break
    except Exception as e:
        print(f"Worker Error ({noise_type}): {e}")


class ParallelEnv:
    def __init__(self, config, num_envs=15):
        self.config = config
        print("Loading thermal data...")
        try:
            mt_data = load_mat()
        except:
            mt_data = None

        print("Pre-calculating projector coefficients...")
        proj = BeamDisturbanceProjector()
        projector_coeffs = proj.get_static_coeffs()

        base_modes = ['jitter', 'maneuver', 'impact', 'mixed', 'thermal']
        self.mode_list = (base_modes * 3)[:num_envs]
        self.num_envs = len(self.mode_list)

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.ps = []
        for i, (work_remote, remote, mode) in enumerate(zip(self.work_remotes, self.remotes, self.mode_list)):
            p = mp.Process(target=worker, args=(work_remote, remote, config, mode, mt_data, projector_coeffs))
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

    def reset(self, seeds=None):
        if seeds is None: seeds = [None] * self.num_envs
        for remote, seed in zip(self.remotes, seeds):
            remote.send(('reset', seed))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_history(self, env_idx):
        self.remotes[env_idx].send(('get_history', None))
        return self.remotes[env_idx].recv()

    def close(self):
        for remote in self.remotes: remote.send(('close', None))
        for p in self.ps: p.join()


def train():
    config = Config()
    start_time = time.time()
    history_rewards = []
    envs = ParallelEnv(config, num_envs=15)
    agent = DDPGAgent(config, num_envs=1)

    trajectory_buffers = [[] for _ in range(envs.num_envs)]
    current_ep_rewards = np.zeros(envs.num_envs)
    env_active_mask = np.ones(envs.num_envs, dtype=bool)
    failure_reasons = [""] * envs.num_envs

    # --- [新增] Phase 0: 专家 PID 预热 ---
    print("\n>>> Phase 0: Expert PID Pre-warming (170, 0, 20)...")
    expert_pid = np.array([170.0, 0.0, 20.0])
    # 调用 config 的归一化函数转换成 Actor 空间 [-1, 1] 的动作
    expert_action_norm = config.normalize_action(expert_pid)

    pre_states = envs.reset()
    for _ in range(config.EPISODE_LENGTH):
        # 所有并行环境都执行专家动作
        actions = np.tile(expert_action_norm, (envs.num_envs, 1))
        next_states, rewards, dones, _ = envs.step(actions)
        for i in range(envs.num_envs):
            # 专家经验也需过滤 NaN 确保质量
            if not np.isnan(next_states[i]).any():
                agent.store_transition(pre_states[i], actions[i], rewards[i], next_states[i], dones[i])
        pre_states = next_states
    print(f">>> Pre-warming finished. Buffer size: {len(agent.memory)}\n")

    print("Start Formal Training (Robust Mode)...")

    for episode in range(config.EPISODES):
        states = envs.reset()
        agent.noise.reset()

        for i in range(envs.num_envs): trajectory_buffers[i] = []
        current_ep_rewards.fill(0)
        env_active_mask.fill(True)
        failure_reasons = [""] * envs.num_envs

        for step in range(config.EPISODE_LENGTH):
            actions = np.zeros((envs.num_envs, config.ACTION_DIM))

            for i in range(envs.num_envs):
                if env_active_mask[i]:
                    if np.isnan(states[i]).any():
                        env_active_mask[i] = False
                        failure_reasons[i] = "State NaN"
                        continue

                    # agent 返回的是 [-1, 1] 的归一化动作
                    action = agent.select_action(states[i], add_noise=True)

                    if np.isnan(action).any():
                        env_active_mask[i] = False
                        failure_reasons[i] = "Action NaN"
                        actions[i] = np.zeros(config.ACTION_DIM)
                    else:
                        actions[i] = action
                else:
                    actions[i] = np.zeros(config.ACTION_DIM)

            next_states, rewards, dones, infos = envs.step(actions)

            # --- [改进] 物理发散与数值审计 ---
            for i in range(envs.num_envs):
                if not env_active_mask[i]: continue

                r_val = rewards[i].item()
                abs_y = infos[i].get('abs_y', 0)

                # 判定条件：NaN 或 物理发散 |Y| > 10000
                is_exploded = np.isnan(next_states[i]).any() or np.isnan(r_val) or abs_y > 10000.0

                if is_exploded:
                    env_active_mask[i] = False
                    failure_reasons[i] = "Diverged (|Y|>1e4)" if abs_y > 10000.0 else "Numerical NaN"

                    # 存入死亡惩罚 (虽然用户不希望 reward 截断，但发散必须有明确的负反馈)
                    clean_next_state = np.zeros_like(states[i])
                    death_penalty = -500.0  # 给予一个显著的负值，不影响正常 reward 范围
                    trajectory_buffers[i].append(
                        (states[i], actions[i], death_penalty, clean_next_state, True)
                    )
                    continue

                trajectory_buffers[i].append((states[i], actions[i], r_val, next_states[i], dones[i]))
                current_ep_rewards[i] += r_val

            # 更新网络：使用延迟更新和多迭代 Critic
            if len(agent.memory) > config.BATCH_SIZE:
                # 遵循 TD3 思想：Critic 迭代次数多于 Actor
                agent.update_networks(update_actor=step%2==0)

            states = next_states

        # === Episode 结束：数据存入与统计 ===
        valid_count = 0
        for i in range(envs.num_envs):
            if len(trajectory_buffers[i]) > 0:
                for t in trajectory_buffers[i]:
                    agent.store_transition(*t)
            if env_active_mask[i]: valid_count += 1

        failed_modes = [f"{envs.mode_list[i]}({failure_reasons[i]})" for i in range(envs.num_envs) if
                        not env_active_mask[i]]

        if valid_count > 0:
            avg_reward = np.mean([current_ep_rewards[i] for i in range(envs.num_envs) if env_active_mask[i]])
        else:
            avg_reward = np.nan
        history_rewards.append(avg_reward)

        print(f"Ep {episode:3d} | Valid: {valid_count:2d}/15 | AvgR: {avg_reward:7.1f} | Fail: {failed_modes[:2]}...")

        # --- 绘图功能保留 ---
        if episode % 10 == 0:
            agent.save_models(MODELS_DIR / f'ddpg_ep_{episode}.pth')
            target_modes = ['impact', 'mixed']
            for target_mode in target_modes:
                try:
                    if target_mode in envs.mode_list:
                        idx = envs.mode_list.index(target_mode)
                        Y, Ref, Kp, Ki, Kd = envs.get_history(idx)
                        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
                        axes[0].plot(Ref, 'k--', label='Ref', alpha=0.5)
                        axes[0].plot(Y, 'b', label='Response')
                        axes[0].set_title(f'Ep {episode} - {target_mode} Response (R={current_ep_rewards[idx]:.1f})')
                        axes[0].legend()
                        axes[0].grid(True)
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
                    print(f"Plotting failed: {e}")

    # === 训练结束绘图保留 ===
    print(f"Training Finished. Total Time: {(time.time() - start_time) / 60:.1f} min")
    agent.save_models(MODELS_DIR / 'ddpg_final_robust.pth')

    plt.figure(figsize=(12, 6))
    rewards_np = np.array(history_rewards)
    plt.plot(rewards_np, color='royalblue', linewidth=1.5, label='Average Reward')
    nan_indices = np.where(np.isnan(rewards_np))[0]
    if len(nan_indices) > 0:
        plt.scatter(nan_indices, [np.nanmin(rewards_np) if not np.all(np.isnan(rewards_np)) else 0] * len(nan_indices),
                    color='red', marker='x', s=20, label='Explosion')
    plt.title('DDPG Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(RESULTS_DIR / 'final_training_raw_curve.png')
    plt.show()

    envs.close()


if __name__ == '__main__':
    train()
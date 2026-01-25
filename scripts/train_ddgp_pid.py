import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
import random

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

    # 1. 环境初始化
    envs = ParallelEnv(config, num_envs=15)

    # 2. Agent 初始化
    # 关键修改：capacity 设大一点给 agent_buffer 用，expert_ratio 初始为 1.0 (全专家)
    agent = DDPGAgent(config, num_envs=1)
    # 我们希望专家数据至少有 150,000 条，这个列表会自动增长，不用担心 capacity
    # capacity 主要是限制 agent_buffer (RL 探索数据) 的大小
    agent.memory.expert_ratio = 1.0

    # =========================================================================
    # PHASE 0: 鲁棒的专家数据采集 (Robust Data Collection)
    # =========================================================================
    TARGET_EXPERT_SAMPLES = 750000  # 目标：采集高质量专家数据
    print(f"\n>>> Phase 0: Robust Data Collection (Target: {TARGET_EXPERT_SAMPLES})...")

    expert_pid = np.array([100.0, 5.0, 25.0])
    expert_action_norm = config.normalize_action(expert_pid)

    current_samples = 0
    round_idx = 0

    # 循环采集，直到存够数据
    while current_samples < TARGET_EXPERT_SAMPLES:
        # 每次循环都生成新的随机种子
        seeds = [random.randint(0, 100000) for _ in range(envs.num_envs)]
        pre_states = envs.reset(seeds=seeds)

        # 每轮只跑 1000-2000 步，然后就重置。
        # 这样可以让池子里充斥着大量的“初始震荡抑制”和“随机冲击恢复”的数据
        # 而不仅仅是平稳运行的数据。
        steps_per_round = 7500

        for _ in range(steps_per_round):
            actions = np.tile(expert_action_norm, (envs.num_envs, 1))
            next_states, rewards, dones, _ = envs.step(actions)

            for i in range(envs.num_envs):
                # 过滤 NaN，只存有效数据
                if not np.isnan(next_states[i]).any():
                    agent.memory.add(pre_states[i], actions[i], rewards[i],
                                     next_states[i], dones[i], is_expert=True)
                    current_samples += 1

            pre_states = next_states

            if current_samples >= TARGET_EXPERT_SAMPLES:
                break

        round_idx += 1
        print(f"  > Collection Round {round_idx}: Total Samples {current_samples}/{TARGET_EXPERT_SAMPLES}")

    print(f">>> Expert Pool Ready. Total Rounds: {round_idx}, Size: {len(agent.memory.expert_buffer)}")

    # =========================================================================
    # PHASE 1: Actor 监督学习 (让飞行员学会开飞机)
    # =========================================================================
    print("\n>>> Phase 1: Actor Behavior Cloning...")
    agent.memory.expert_ratio = 1.0
    for i in range(5000):
        loss = agent.update_actor_supervised(batch_size=256)
        if i % 1000 == 0:
            print(f"  [BC] Iter {i} | Actor Loss: {loss:.6f}")

    # 关键：BC 结束后立即硬同步 Actor Target
    agent.hard_update(agent.actor_target, agent.actor)

    # =========================================================================
    # PHASE 2: Critic 价值预热 (让指挥部学会评分)
    # =========================================================================
    print("\n>>> Phase 2: Critic Value Warm-up...")
    # 依然只使用专家数据，但这次是练 Critic
    for i in range(20000):
        c_loss = agent.pretrain_critic(batch_size=256)
        if i % 1000 == 0:
            print(f"  [Warmup] Iter {i} | Critic Loss: {c_loss:.6f}")

    # 再次硬同步所有网络，确保正式开跑前 Target = Local
    agent.hard_update(agent.actor_target, agent.actor)
    agent.hard_update(agent.critic_target, agent.critic)
    print(">>> All Networks Pre-trained and Synced.")

    # =========================================================================
    # PHASE 3: 进入 RL 训练
    # =========================================================================
    print("\n>>> Phase 2: Start RL Fine-tuning...")
    agent.memory.expert_ratio = 0.5  # 恢复混合采样

    # --- 正式训练 ---
    for episode in range(config.EPISODES):
        states = envs.reset()
        agent.noise.reset()

        # 每一轮开始前重置统计变量
        trajectory_buffers = [[] for _ in range(envs.num_envs)]
        current_ep_rewards = np.zeros(envs.num_envs)
        env_active_mask = np.ones(envs.num_envs, dtype=bool)
        failure_reasons = [""] * envs.num_envs  # 确保在这里定义

        for step in range(config.EPISODE_LENGTH):
            # 1. 动作生成
            actions = np.zeros((envs.num_envs, config.ACTION_DIM))
            for i in range(envs.num_envs):
                if env_active_mask[i]:
                    action = agent.select_action(states[i], add_noise=True)
                    if np.isnan(action).any():
                        env_active_mask[i] = False
                        failure_reasons[i] = "Action NaN"
                    else:
                        actions[i] = action

            # 2. 环境步进
            next_states, rewards, dones, infos = envs.step(actions)

            # 3. 实时审计
            for i in range(envs.num_envs):
                if not env_active_mask[i]: continue

                r_val = rewards[i].item()
                abs_y = infos[i].get('abs_y', 0)

                # 发散判定
                is_diverged = abs_y > 10000.0
                is_nan = np.isnan(next_states[i]).any()

                if is_diverged or is_nan:
                    env_active_mask[i] = False
                    failure_reasons[i] = "Diverged" if is_diverged else "NextState NaN"
                    # 存入一笔带有死亡惩罚的终止经验
                    trajectory_buffers[i].append((states[i], actions[i], -500.0, np.zeros_like(states[i]), True))
                    continue

                # 正常经验暂存
                trajectory_buffers[i].append((states[i], actions[i], r_val, next_states[i], dones[i]))
                current_ep_rewards[i] += r_val

            # 4. 网络更新
            if len(agent.memory.agent_buffer) > config.BATCH_SIZE:
                agent.update_networks(update_actor=step % 2 == 0)

            states = next_states

        # === Episode 结束：结算数据并打印失败原因 ===
        valid_count = 0
        for i in range(envs.num_envs):
            # 将暂存的轨迹刷入 agent_buffer (is_expert=False)
            if trajectory_buffers[i]:
                for t in trajectory_buffers[i]:
                    agent.memory.add(*t, is_expert=False)
            if env_active_mask[i]:
                valid_count += 1

        failed_modes = [f"{envs.mode_list[i]}({failure_reasons[i]})" for i in range(envs.num_envs) if
                        not env_active_mask[i]]

        if valid_count > 0:
            avg_reward = np.mean([current_ep_rewards[i] for i in range(envs.num_envs) if env_active_mask[i]])
        else:
            avg_reward = np.nan
        history_rewards.append(avg_reward)

        print(f"Ep {episode:3d} | Valid: {valid_count:2d}/15 | AvgR: {avg_reward:7.1f} | Fail: {failed_modes[:2]}...")

        # --- 绘图功能保留 ---
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
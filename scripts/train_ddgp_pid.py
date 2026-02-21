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
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
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
                action = data  # 归一化动作
                next_state, reward, done, info = env.step(action)

                # === 修改：用 env.current_step 取索引，而不是 env.state_space.current_step ===
                idx = max(env.current_step - 1, 0)
                current_y = float(env.state_space.Y[idx]) if hasattr(env.state_space, "Y") else 0.0
                info['abs_y'] = abs(current_y)

                remote.send((next_state, reward, done, info))

            elif cmd == 'reset':
                env = make_env(data)
                state = env.reset()
                remote.send(state)

            elif cmd == 'get_history':
                # 兼容老版环境：kp/ki/kd 是数组字段
                remote.send((
                    env.state_space.Y,
                    getattr(env.state_space, "Reference", np.zeros(env.episode_length)),
                    env.state_space.kp,
                    env.state_space.ki,
                    env.state_space.kd
                ))

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
        except Exception:
            mt_data = None

        print("Pre-calculating projector coefficients...")
        proj = BeamDisturbanceProjector()
        projector_coeffs = proj.get_static_coeffs()

        base_modes = ['jitter', 'maneuver', 'impact', 'mixed', 'thermal']
        self.mode_list = (base_modes * 3)[:num_envs]
        self.num_envs = len(self.mode_list)

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.ps = []
        for work_remote, remote, mode in zip(self.work_remotes, self.remotes, self.mode_list):
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

    # 1) 环境初始化
    envs = ParallelEnv(config, num_envs=15)

    # 2) Agent 初始化
    agent = DDPGAgent(config, num_envs=1)
    agent.memory.expert_ratio = 1.0

    # ================================
    # 回滚机制参数（新增）
    # ================================
    ROLLBACK_PATH = MODELS_DIR / "_checkpoint_last_good.pth"
    # 爆炸后恢复更稳：降低探索噪声一段时间 / 暂停 actor 更新一段时间
    EXPLODE_COOLDOWN_EP = 5
    cooldown_left = 0

    # 你可以把阈值改成 1000（你说正常<300）
    DIVERGE_Y_THRESH = 1000.0

    # =========================================================================
    # PHASE 0: 专家数据采集
    # =========================================================================
    TARGET_EXPERT_SAMPLES = 750000
    print(f"\n>>> Phase 0: Robust Data Collection (Target: {TARGET_EXPERT_SAMPLES})...")

    expert_pid = np.array([100.0, 5.0, 25.0], dtype=float)
    expert_action_norm = config.normalize_action(expert_pid)

    current_samples = 0
    round_idx = 0

    while current_samples < TARGET_EXPERT_SAMPLES:
        seeds = [random.randint(0, 100000) for _ in range(envs.num_envs)]
        pre_states = envs.reset(seeds=seeds)

        steps_per_round = 7500

        for _ in range(steps_per_round):
            actions = np.tile(expert_action_norm, (envs.num_envs, 1))
            next_states, rewards, dones, _ = envs.step(actions)

            for i in range(envs.num_envs):
                if not np.isnan(next_states[i]).any():
                    agent.memory.add(pre_states[i], actions[i], float(rewards[i]),
                                     next_states[i], bool(dones[i]), is_expert=True)
                    current_samples += 1

            pre_states = next_states
            if current_samples >= TARGET_EXPERT_SAMPLES:
                break

        round_idx += 1
        print(f"  > Collection Round {round_idx}: Total Samples {current_samples}/{TARGET_EXPERT_SAMPLES}")

    print(f">>> Expert Pool Ready. Total Rounds: {round_idx}, Size: {len(agent.memory.expert_buffer)}")

    # =========================================================================
    # PHASE 1: Actor BC
    # =========================================================================
    print("\n>>> Phase 1: Actor Behavior Cloning...")
    agent.memory.expert_ratio = 1.0
    for i in range(5000):
        loss = agent.update_actor_supervised(batch_size=256)
        if i % 1000 == 0:
            print(f"  [BC] Iter {i} | Actor Loss: {loss:.6f}")

    agent.hard_update(agent.actor_target, agent.actor)

    # 保存“最后良好”checkpoint（新增）
    agent.save_models(ROLLBACK_PATH)

    # =========================================================================
    # PHASE 2: Critic 预热（固定迭代次数）
    # =========================================================================
    print("\n>>> Phase 2: Critic Value Warm-up...")
    for i in range(10000):
        c_loss = agent.pretrain_critic(batch_size=256)
        if (i+1) % 1000 == 0:
            print(f"  [Warmup] Iter {i+1} | Critic Loss: {c_loss:.6f}")

    agent.hard_update(agent.actor_target, agent.actor)
    agent.hard_update(agent.critic_target, agent.critic)
    print(">>> All Networks Pre-trained and Synced.")

    # 更新回滚点（新增）
    agent.save_models(ROLLBACK_PATH)

    # =========================================================================
    # PHASE 3: RL 训练
    # =========================================================================
    print("\n>>> Phase 3: Start RL Fine-tuning...")
    agent.memory.expert_ratio = 0.5

    for episode in range(config.EPISODES):
        states = envs.reset()
        agent.noise.reset()

        # ===== 新增：爆炸冷却期策略 =====
        # 冷却期内不更新 actor（只更新 critic），并降低探索噪声
        update_actor_flag = True
        if cooldown_left > 0:
            update_actor_flag = False
            cooldown_left -= 1
            # 如果你 OU 噪声支持动态 sigma，可在这里调小；否则只能接受固定 sigma
            # 这里不强改 agent 代码，保持接口不变

        trajectory_buffers = [[] for _ in range(envs.num_envs)]
        current_ep_rewards = np.zeros(envs.num_envs, dtype=float)
        env_active_mask = np.ones(envs.num_envs, dtype=bool)
        failure_reasons = [""] * envs.num_envs

        episode_exploded = False  # 新增：用于回滚判定

        for step in range(config.EPISODE_LENGTH):
            actions = np.zeros((envs.num_envs, config.ACTION_DIM), dtype=float)

            # 1) 动作生成
            for i in range(envs.num_envs):
                if env_active_mask[i]:
                    action = agent.select_action(states[i], add_noise=True)
                    if np.isnan(action).any():
                        env_active_mask[i] = False
                        failure_reasons[i] = "Action NaN"
                        episode_exploded = True
                    else:
                        actions[i] = action

            # 2) 环境步进
            next_states, rewards, dones, infos = envs.step(actions)

            # 3) 审计
            for i in range(envs.num_envs):
                if not env_active_mask[i]:
                    continue

                r_val = float(rewards[i])
                abs_y = float(infos[i].get('abs_y', 0.0))

                is_diverged = abs_y > DIVERGE_Y_THRESH
                is_nan = np.isnan(next_states[i]).any()

                if is_diverged or is_nan:
                    env_active_mask[i] = False
                    failure_reasons[i] = "Diverged" if is_diverged else "NextState NaN"
                    episode_exploded = True
                    # 终止经验先暂存，但注意：爆炸 episode 我们后面会整体丢弃，不写入 memory
                    trajectory_buffers[i].append((states[i], actions[i], -500.0, np.zeros_like(states[i]), True))
                    continue

                trajectory_buffers[i].append((states[i], actions[i], r_val, next_states[i], bool(dones[i])))
                current_ep_rewards[i] += r_val

            # 4) 网络更新
            if len(agent.memory.agent_buffer) > config.BATCH_SIZE:
                agent.update_networks(update_actor=(update_actor_flag and (step % 2 == 0)))

            states = next_states

            # 如果已经爆炸，没必要继续推进（新增：提前结束本 episode）
            if episode_exploded:
                break

        # ===== Episode 结束处理 =====
        valid_count = int(np.sum(env_active_mask))

        # 新增：更严格的“爆炸判定”
        # - 任一 env 出现 NaN/发散（episode_exploded）
        # - 或 avg_reward 不是有限值
        if valid_count > 0:
            avg_reward = np.mean([current_ep_rewards[i] for i in range(envs.num_envs) if env_active_mask[i]])
        else:
            avg_reward = np.nan

        if (episode_exploded or (not np.isfinite(avg_reward))):
            # ------------------------------
            # 回滚策略（新增，核心）
            # ------------------------------
            print(f"[Rollback] Ep {episode} exploded. Reverting to last good checkpoint and discarding episode data.")
            try:
                agent.load_models(ROLLBACK_PATH)
            except Exception as e:
                print(f"[Rollback] Failed to load checkpoint: {e}")

            # 丢弃本回合所有 trajectory_buffers（不写入 agent_buffer） -> 避免污染
            history_rewards.append(np.nan)

            # 进入冷却期：先只训 critic 不训 actor，降低继续爆炸概率
            cooldown_left = EXPLODE_COOLDOWN_EP

            # 保存失败信息
            failed_modes = [f"{envs.mode_list[i]}({failure_reasons[i]})"
                            for i in range(envs.num_envs) if not env_active_mask[i]]
            print(f"Ep {episode:3d} | Valid: {valid_count:2d}/15 | AvgR:   NaN | Fail: {failed_modes[:3]}...")
            continue

        # 正常回合：刷入 agent_buffer
        for i in range(envs.num_envs):
            if trajectory_buffers[i]:
                for t in trajectory_buffers[i]:
                    agent.memory.add(*t, is_expert=False)

        failed_modes = [f"{envs.mode_list[i]}({failure_reasons[i]})"
                        for i in range(envs.num_envs) if not env_active_mask[i]]

        history_rewards.append(avg_reward)

        print(f"Ep {episode:3d} | Valid: {valid_count:2d}/15 | AvgR: {avg_reward:9.2f} | Fail: {failed_modes[:2]}...")

        # 正常回合：更新“最后良好回滚点”（新增）
        agent.save_models(ROLLBACK_PATH)

        # --- 你原来的绘图功能保留 ---
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

    # 训练结束
    print(f"Training Finished. Total Time: {(time.time() - start_time) / 60:.1f} min")
    agent.save_models(MODELS_DIR / 'ddpg_final_robust.pth')

    # 训练曲线（原样保留）
    plt.figure(figsize=(12, 6))
    rewards_np = np.array(history_rewards, dtype=float)
    plt.plot(rewards_np, linewidth=1.5, label='Average Reward')

    nan_indices = np.where(~np.isfinite(rewards_np))[0]
    if len(nan_indices) > 0:
        y_anchor = np.nanmin(rewards_np[np.isfinite(rewards_np)]) if np.any(np.isfinite(rewards_np)) else 0.0
        plt.scatter(nan_indices, np.full(len(nan_indices), y_anchor), marker='x', s=20, label='Explosion')

    plt.title('DDPG Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(RESULTS_DIR / 'final_training_raw_curve.png')
    plt.show()

    envs.close()


if __name__ == '__main__':
    # Windows 多进程建议
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    train()
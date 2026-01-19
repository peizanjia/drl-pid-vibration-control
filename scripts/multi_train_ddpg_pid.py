import numpy as np
import matplotlib.pyplot as plt
import os
import time
import multiprocessing as mp
from pathlib import Path
import torch

# 假设这些是你项目中的模块，保持引用
from src.environments.environment_2 import PIDControlEnvironment
from src.agents.ddpg_agent import DDPGAgent
from config.config import Config
from src.utils.utils import set_seed, create_noise_data
# 必须确保 StateSpace 可以被 pickling，或者在 worker 内部重新 import
from src.solvers.state_space_2 import StateSpace

# 防止 OpenMP 在多进程中冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ==========================================
# 1. 路径与配置管理 (修复 Task 0)
# ==========================================
# 获取当前脚本的绝对路径 -> root/scripts/train.py -> root/
ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / 'models'
RESULTS_DIR = ROOT_DIR / 'results'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================
# 2. 并行环境工作流 (修复 Task 1 & 3)
# ==========================================
def worker_process(remote, parent_remote, config, noise_type, worker_id):
    """
    运行在独立进程中的环境实例
    """
    parent_remote.close()

    # 在子进程中重新生成噪声数据，确保独立性
    tn = config.EPISODE_LENGTH
    # 假设 create_noise_data 已经修改为接受 option 参数
    # Task 1: 这里根据分配的 noise_type 生成特定数据
    noise_data = create_noise_data(tn, option=noise_type)

    # 初始化状态空间
    state_space = StateSpace(config.SYSTEM_CONFIG, noise_data, dt=config.DT, tn=tn)

    # 初始化环境
    env = PIDControlEnvironment(config)
    env.set_state_space(state_space)

    # 设置不同的随机种子，防止所有环境走出一样的随机步
    np.random.seed(worker_id + int(time.time()))

    try:
        while True:
            cmd, data = remote.recv()

            if cmd == 'step':
                action = data
                next_state, reward, done, info = env.step(action)
                # 如果 done，自动 reset (这是 VectorEnv 的标准做法)
                if done:
                    # 对于你的 PID 环境，reset 可能意味着重置状态空间索引
                    # 这里的 reset 逻辑取决于你的环境实现，通常返回初始状态
                    next_state = env.reset()
                remote.send((next_state, reward, done, info))

            elif cmd == 'reset':
                state = env.reset()
                remote.send(state)

            elif cmd == 'get_history':
                # 获取用于绘图的数据
                remote.send((env.state_space.Y, env.state_space.Reference))

            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError(f"Worker received unknown command: {cmd}")
    except Exception as e:
        print(f"Worker {worker_id} (Noise: {noise_type}) failed: {e}")
        remote.close()


class ParallelEnvManager:
    """
    管理多个并行环境的主控类
    """

    def __init__(self, config, noise_types):
        self.config = config
        self.noise_types = noise_types
        self.num_envs = len(noise_types)

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.ps = []

        for i, (work_remote, remote, noise_type) in enumerate(zip(self.work_remotes, self.remotes, noise_types)):
            p = mp.Process(target=worker_process,
                           args=(work_remote, remote, config, noise_type, i))
            p.daemon = True  # 主进程死掉子进程也随之关闭
            p.start()
            self.ps.append(p)
            work_remote.close()  # 父进程不需要 work_remote

    def step(self, actions):
        # 发送 Action 到所有 Worker
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        # 接收结果
        results = [remote.recv() for remote in self.remotes]
        # 解包: results is list of (next_state, reward, done, info)
        next_states, rewards, dones, infos = zip(*results)
        return np.stack(next_states), np.stack(rewards), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_history(self, env_idx=0):
        """只获取指定环境的历史数据用于绘图"""
        self.remotes[env_idx].send(('get_history', None))
        return self.remotes[env_idx].recv()

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()


# ==========================================
# 3. 主训练流程
# ==========================================
def train_ddpg_pid_parallel():
    # 任务 3: 并行化逻辑
    # 我们定义 5 种噪声类型，如果需要更多核心，可以复制这个列表
    # 例如: noise_scenarios = ['jitter', 'maneuver', 'impact', 'mixed', 'thermal'] * 2 (共10核)
    noise_scenarios = ['jitter', 'maneuver', 'impact', 'mixed', 'thermal']

    config = Config()

    # 更新 Config 以适应并行环境的 Batch Size
    # 注意: 如果你的 Replay Buffer 是按单条存的，需要调整 add 逻辑

    # 初始化并行环境管理器
    print(f"初始化并行环境，使用 {len(noise_scenarios)} 个核心...")
    env_manager = ParallelEnvManager(config, noise_scenarios)

    # 创建 DDPG 智能体
    agent = DDPGAgent(config)

    episode_rewards = []
    critic_losses = []
    actor_losses = []

    print("开始训练 DDPG 自适应 PID 控制器 (Parallel Mode)...")
    print(f"PID参数范围: Kp={config.KP_RANGE}, Ki={config.KI_RANGE}, Kd={config.KD_RANGE}")

    start_time = time.time()
    total_steps = 0  # 全局步数

    for episode in range(config.EPISODES):
        # 1. Reset 所有环境
        states = env_manager.reset()  # shape: (num_envs, state_dim)
        agent.noise.reset()  # 重置 OU 噪声

        # 用于统计本回合所有环境的总奖励
        batch_episode_reward = np.zeros(env_manager.num_envs)

        # 临时存储 Loss
        ep_critic_loss = []
        ep_actor_loss = []

        # -------------------------------------------------------
        # Episode Loop (Vectorized)
        # -------------------------------------------------------
        for step in range(config.EPISODE_LENGTH):

            # 2. Action Selection (Batch)
            if total_steps < config.WARMUP_STEPS:
                actions = np.random.uniform(-1, 1, (env_manager.num_envs, config.ACTION_DIM))
            else:
                # 需确保 agent.select_action 能处理 batch input，如果不能，需修改 agent 或用循环
                # 这里假设 agent 接收 batch state 并返回 batch action
                # 如果你的 agent 只能处理单个，需要这里做个循环：
                actions = np.array([agent.select_action(s) for s in states])

            # 3. Environment Step (Parallel)
            next_states, rewards, dones, infos = env_manager.step(actions)

            # 4. 存储经验 (由于 Buffer 通常是串行的，这里循环存入)
            # 也可以优化为 Batch Add，取决于你的 Buffer 实现
            for i in range(env_manager.num_envs):
                agent.store_transition(states[i], actions[i], rewards[i], next_states[i], dones[i])
                batch_episode_reward[i] += rewards[i]

            # 5. 更新网络 (DDPG Update)
            # 任务 2 修正: 严格控制 Actor 更新频率
            # 每收集一步数据（实际上是 num_envs 步），更新网络
            if len(agent.memory) > config.BATCH_SIZE:

                # 策略 1: 每一步都更新 Critic，但每 d 步更新 Actor (Delayed Update)
                # 修正后的逻辑：Critc 更新次数 > Actor 更新次数
                c_loss, a_loss = agent.update_networks()  # 假设内部逻辑已处理 Update

                # 如果 agent.update_networks 内部没有分别控制，这里手动写:
                # 这里我无法直接修改你的 agent 代码，但给出一个典型的逻辑控制示例：
                # if total_steps % policy_freq == 0:
                #     agent.update_actor()
                # agent.update_critic()

                if c_loss is not None:
                    ep_critic_loss.append(c_loss)
                if a_loss is not None:
                    ep_actor_loss.append(a_loss)

            states = next_states
            total_steps += env_manager.num_envs

            # 这里的 done 处理比较微妙，因为是固定长度 episode，通常不用 break
            # 如果某个环境提前 done，ParallelEnvManager 会自动 reset，我们继续训练即可
            if np.any(dones):
                # 在固定长度任务中，通常不需要因为一个环境 done 就全部退出
                # 除非所有环境都 done，或者达到了 EPISODE_LENGTH
                pass

        # -------------------------------------------------------
        # Episode End Handling
        # -------------------------------------------------------

        avg_ep_reward = np.mean(batch_episode_reward)
        avg_c_loss = np.mean(ep_critic_loss) if ep_critic_loss else 0
        avg_a_loss = np.mean(ep_actor_loss) if ep_actor_loss else 0

        episode_rewards.append(avg_ep_reward)
        critic_losses.append(avg_c_loss)
        actor_losses.append(avg_a_loss)

        if episode % 10 == 0:
            print(f"Episode {episode} | Avg Reward (across {len(noise_scenarios)} envs): {avg_ep_reward:.4e} | "
                  f"Critic Loss: {avg_c_loss:.4e}")

            # 保存模型
            agent.save_models(MODELS_DIR / f'ddpg_pid_episode_{episode}.pth')

            # 绘图: 只取第 0 个环境（比如 Jitter）的数据做展示
            # 注意：多进程中无法直接获取对象属性，必须通过 Pipe 通信
            try:
                Y_data, Ref_data = env_manager.get_history(env_idx=0)  # 取 Jitter 场景

                plt.figure(figsize=(12, 6))
                plt.plot(Y_data, label=f'Output (Noise: {noise_scenarios[0]})', color='blue')
                if Ref_data is not None:
                    plt.plot(Ref_data, label='Reference', color='red', linestyle='--')
                plt.title(f'Response Analysis - Episode {episode}', fontsize=16)
                plt.xlabel('Time Steps')
                plt.ylabel('Response')
                plt.legend()
                plt.grid(True)
                plt.savefig(RESULTS_DIR / f'response_ep_{episode}.png')
                plt.close()
            except Exception as e:
                print(f"绘图失败: {e}")

    # 结束清理
    env_manager.close()

    print(f"训练完成! 总耗时: {time.time() - start_time:.2f}s")
    agent.save_models(MODELS_DIR / 'ddpg_pid_final.pth')

    # 绘制最终训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Average Episode Reward (Parallel Training)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig(RESULTS_DIR / 'training_curve.png')


if __name__ == "__main__":
    # Windows 下多进程必须在 if __name__ == "__main__": 下运行
    train_ddpg_pid_parallel()
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from src.environments.environment_2 import PIDControlEnvironment
from src.agents.ddpg_agent import DDPGAgent
from config.config import Config
from src.utils.utils import set_seed, create_noise_data
from src.solvers.state_space_2 import StateSpace  # 确保 StateSpace 已导入

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train_ddpg_pid_efficient():
    """训练DDPG自适应PID控制器（高效版）"""

    # 1. --- 种子和配置 ---
    # 在脚本开始时设置一次种子，用于复现整个实验（例如，网络初始化）
    set_seed(42)
    config = Config()

    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 2. --- 初始化 Agent 和环境占位符 ---
    # 注意：环境现在将在循环内部创建

    # 创建DDPG智能体 (Agent 只需要创建一次)
    agent = DDPGAgent(config)

    # (环境和 StateSpace 将在循环内创建)
    env = PIDControlEnvironment(config)

    # 训练统计
    episode_rewards = []
    episode_costs = []
    critic_losses = []
    actor_losses = []

    print("开始高效训练DDPG自适应PID控制器...")
    print(f"状态维度: {config.STATE_DIM}, 动作维度: {config.ACTION_DIM}")
    print(f"设备: {config.DEVICE}")
    print(f"训练模式: 逐-回合-更新 (Episodic Update)")

    start_time = time.time()
    total_steps = 0

    for episode in range(config.EPISODES):

        # ===================================================================
        # 【修改 1】在每回合开始时创建新的随机噪声和环境
        # 这确保了 Agent 面对的是随机的、非重复的扰动
        # ===================================================================
        tn = config.EPISODE_LENGTH
        # 每次循环调用 create_noise_data，RNG 状态不同，噪声也不同
        noise_data = create_noise_data(tn)
        state_space = StateSpace(config.SYSTEM_CONFIG, noise_data, dt=config.DT, tn=tn)
        env.set_state_space(state_space)

        state = env.reset()
        agent.noise.reset()  # 重置 Agent 的探索噪声

        episode_reward = 0
        steps_in_episode = 0

        # ===================================================================
        # 【修改 2】阶段一：数据收集 (内循环)
        # 在此阶段，我们只运行环境和存储数据，不更新网络
        # ===================================================================
        for step in range(config.EPISODE_LENGTH):
            # 选择动作（归一化的PID参数）
            if total_steps < config.WARMUP_STEPS:
                action = np.random.uniform(-1, 1, config.ACTION_DIM)
            else:
                action = agent.select_action(state)

            # 与环境交互
            next_state, reward, done, info = env.step(action)

            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            total_steps += 1
            steps_in_episode += 1

            if done:
                break

        # ===================================================================
        # 【修改 2】阶段二：集中学习 (外循环)
        # 在回合结束后，我们集中更新网络 N 次（N = 刚刚收集的步数）
        # ===================================================================
        episode_critic_loss = 0
        episode_actor_loss = 0
        update_count = 0

        # 只有在总步数超过热身（且缓冲区足够大）时才开始学习
        if total_steps > config.WARMUP_STEPS:
            # print(f"[Ep {episode}] 收集完成, 开始 {steps_in_episode} 次更新...")
            # (GPU 将在此处满载)
            for _ in range(steps_in_episode):
                critic_loss, actor_loss = agent.update_networks()
                if critic_loss is not None:
                    episode_critic_loss += critic_loss
                    episode_actor_loss += actor_loss
                    update_count += 1

        # 计算平均损失
        avg_critic_loss = (episode_critic_loss / update_count) if update_count > 0 else 0
        avg_actor_loss = (episode_actor_loss / update_count) if update_count > 0 else 0

        # 获取性能指标
        metrics = env.get_performance_metrics()

        episode_rewards.append(episode_reward)
        episode_costs.append(metrics['total_cost'])
        critic_losses.append(avg_critic_loss)
        actor_losses.append(avg_actor_loss)

        # 打印进度
        avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
        avg_cost = np.mean(episode_costs[-50:]) if len(episode_costs) >= 50 else np.mean(episode_costs)
        print(f"Episode {episode}, Reward: {episode_reward[0]:.4e}, "
              f"Avg Reward: {avg_reward:.4e}, Cost: {metrics['total_cost'][0]:.4e}, "
              f"Avg Cost: {avg_cost:.4e}, Updates: {update_count}")

        if episode % 1 == 0:
            # 保存模型
            agent.save_models(f'models/ddpg_pid_episode_{episode}.pth')

    # 训练结束
    training_time = time.time() - start_time
    print(f"训练完成! 总时间: {training_time:.2f}秒, 总步数: {total_steps}")

    # 保存最终模型
    agent.save_models('models/ddpg_pid_final.pth')

    # 绘制训练曲线 (与原版相同)
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(episode_costs)
    plt.title('Episode Costs')
    plt.xlabel('Episode')
    plt.ylabel('Total Cost')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(critic_losses, label='Critic Loss')
    plt.plot(actor_losses, label='Actor Loss')
    plt.title('Training Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    window = 50
    # 确保有足够的数据用于滑动平均
    if len(episode_rewards) > window:
        moving_avg = [np.mean(episode_rewards[i:i + window]) for i in range(len(episode_rewards) - window)]
        plt.plot(range(window, len(episode_rewards)), moving_avg)
    else:
        plt.plot(episode_rewards, label='Rewards (less than window)')  # 如果数据不够，直接画
    plt.title('Moving Average Reward (Window=50)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/training_curves_pid.png')
    plt.show()

    # 保存训练数据
    np.savez('results/training_data_pid.npz',
             rewards=episode_rewards,
             costs=episode_costs,
             critic_losses=critic_losses,
             actor_losses=actor_losses)

    return agent, env


if __name__ == "__main__":
    train_ddpg_pid_efficient()
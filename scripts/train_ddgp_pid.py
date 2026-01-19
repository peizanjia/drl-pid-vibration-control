import numpy as np
import matplotlib.pyplot as plt
import os
import time
from src.environments.environment_2 import PIDControlEnvironment
from src.agents.ddpg_agent import DDPGAgent
from config.config import Config
from src.utils.utils import set_seed, create_noise_data

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train_ddpg_pid():
    """训练DDPG自适应PID控制器"""

    # 设置随机种子
    # set_seed(42)

    # 初始化配置和环境
    config = Config()

    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 创建噪声数据
    tn = config.EPISODE_LENGTH
    noise_data = create_noise_data(tn)

    # 创建StateSpace实例
    from src.solvers.state_space_2 import StateSpace
    state_space = StateSpace(config.SYSTEM_CONFIG, noise_data, dt=config.DT, tn=tn)

    # 创建强化学习环境
    env = PIDControlEnvironment(config)
    env.set_state_space(state_space)

    # 创建DDPG智能体
    agent = DDPGAgent(config)

    # 训练统计
    episode_rewards = []
    episode_costs = []
    critic_losses = []
    actor_losses = []

    print("开始训练DDPG自适应PID控制器...")
    print(f"状态维度: {config.STATE_DIM}, 动作维度: {config.ACTION_DIM}")
    print(f"设备: {config.DEVICE}")
    print(f"PID参数范围: Kp={config.KP_RANGE}, Ki={config.KI_RANGE}, Kd={config.KD_RANGE}")

    start_time = time.time()
    total_steps = 0

    for episode in range(config.EPISODES):
        state = env.reset()
        agent.noise.reset()

        episode_reward = 0
        episode_critic_loss = 0
        episode_actor_loss = 0
        update_count = 0

        for step in range(config.EPISODE_LENGTH):
            # 选择动作（归一化的PID参数）
            if total_steps < config.WARMUP_STEPS:
                # 热身阶段使用随机动作探索
                action = np.random.uniform(-1, 1, config.ACTION_DIM)
            else:
                action = agent.select_action(state)

            # 与环境交互
            next_state, reward, done, info = env.step(action)

            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)

            # 更新网络
            critic_loss, actor_loss = agent.update_networks()
            if critic_loss is not None:
                episode_critic_loss += critic_loss
                episode_actor_loss += actor_loss
                update_count += 1

            state = next_state
            episode_reward += reward
            total_steps += 1

            if done:
                break

        # 计算平均损失
        if update_count > 0:
            avg_critic_loss = episode_critic_loss / update_count
            avg_actor_loss = episode_actor_loss / update_count
        else:
            avg_critic_loss = 0
            avg_actor_loss = 0

        # 获取性能指标
        metrics = env.get_performance_metrics()

        episode_rewards.append(episode_reward)
        episode_costs.append(metrics['total_cost'])
        critic_losses.append(avg_critic_loss)
        actor_losses.append(avg_actor_loss)

        # 打印进度
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            avg_cost = np.mean(episode_costs[-50:]) if len(episode_costs) >= 50 else np.mean(episode_costs)
            print(f"Episode {episode}, Reward: {episode_reward[0]:.4e}, "
                  f"Avg Reward: {avg_reward:.4e}, Cost: {metrics['total_cost'][0]:.4e}, "
                  f"Avg Cost: {avg_cost:.4e}")

            # 保存模型
            agent.save_models(f'models/ddpg_pid_episode_{episode}.pth')








            #test plot
            # 1. 准备横坐标数据（索引值）
            x_indices = np.arange(len(env.state_space.Y))

            # 2. 创建图表
            plt.figure(figsize=(12, 6))

            # 3. 绘制曲线
            # 以索引值为横坐标，Y 数组元素为纵坐标
            plt.plot(x_indices, env.state_space.Y, label='Y 数据值', color='blue', linewidth=1.5)

            # 4. 设置图表属性
            plt.title('电压时间', fontsize=16)
            plt.xlabel('时间', fontsize=14)
            plt.ylabel('电压', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()

            # 5. 保存图表
            plt.savefig(f'results/voltage_plot_ep_{episode}.png')
            plt.close('all')











    # 训练结束
    training_time = time.time() - start_time
    print(f"训练完成! 总时间: {training_time:.2f}秒, 总步数: {total_steps}")

    # 保存最终模型
    agent.save_models('models/ddpg_pid_final.pth')

    # 绘制训练曲线
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
    # 计算滑动平均奖励
    window = 50
    moving_avg = [np.mean(episode_rewards[i:i + window]) for i in range(len(episode_rewards) - window)]
    plt.plot(range(window, len(episode_rewards)), moving_avg)
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
    train_ddpg_pid()
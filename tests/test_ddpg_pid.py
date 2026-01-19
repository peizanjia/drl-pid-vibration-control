import matplotlib.pyplot as plt
import os
import time
import pandas as pd  # 新增：用于专业的数据保存
import numpy as np  # 新增：用于数值计算
from src.environments.environment_2 import PIDControlEnvironment
from src.agents.ddpg_agent import DDPGAgent
from config.config import Config
from src.utils.utils import set_seed, create_noise_data
from src.solvers.state_space_2 import StateSpace


def test_ddpg_pid_consolidated():
    """
    批处理测试 DDPG 模型，并生成符合申报要求的源数据文件 (CSV) 和汇总表格。
    """

    set_seed(int(time.time()))
    config = Config()

    # 1. 初始化环境
    tn = config.EPISODE_LENGTH
    noise_data = create_noise_data(tn)
    state_space = StateSpace(config.SYSTEM_CONFIG, noise_data, dt=config.DT, tn=tn)
    env = PIDControlEnvironment(config)
    env.set_state_space(state_space)
    agent = DDPGAgent(config)

    # 2. 定义测试案例
    test_case = {
        'name': 'Random_Noise_Rejection',
        'initial_state': [0, 0, 0, 0],
        'excitation': 'random'
    }

    # 📁 创建保存目录
    # "source_data" 文件夹专门存放原始 CSV 数据，对应成熟度 5 级要求
    data_dir = 'source_data'
    img_dir = 'results_images'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # 用于汇总所有 Episode 表现的列表
    summary_list = []

    # 3. 循环测试
    # 建议根据实际模型数量调整 range，这里假设测试前 50 个或者特定的几个
    episodes_to_test = range(100)

    for episode_idx in episodes_to_test:
        model_path = f'models/ddpg_pid_episode_{episode_idx}.pth'

        if not os.path.exists(model_path):
            continue  # 模型不存在则跳过，不报错

        print(f"Testing Model: Episode {episode_idx}...")

        try:
            agent.load_models(model_path)
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            continue

        # 重置环境
        state = env.reset(test_case['initial_state'])

        # 数据容器
        outputs = []
        kp_values, ki_values, kd_values = [], [], []
        time_steps = []
        rewards = []  # 记录单步奖励

        # 4. 模拟运行
        for step in range(config.EPISODE_LENGTH):
            # 测试时通常关闭噪声 (add_noise=False) 以评估确定性策略
            action = agent.select_action(state, add_noise=False)
            next_state, reward, done, info = env.step(action)

            # 收集数据
            current_time = step * config.DT
            outputs.append(info['output'])
            kp_values.append(info['kp'])
            ki_values.append(info['ki'])
            kd_values.append(info['kd'])
            time_steps.append(current_time)
            rewards.append(reward)

            state = next_state
            if done:
                break

        # 获取性能指标
        metrics = env.get_performance_metrics()
        total_cost_value = metrics['total_cost'].item()

        # ==========================================
        # 🟢 关键修改 1：保存单次实验的源数据 (CSV)
        # ==========================================
        # 这就是你需要提交的“源数据文件”
        df_episode = pd.DataFrame({
            'Time (s)': time_steps,
            'Output': outputs,
            'Kp': kp_values,
            'Ki': ki_values,
            'Kd': kd_values,
            'Step_Reward': rewards
        })

        # 保存 CSV，文件名包含 Cost 方便后续筛选
        csv_filename = f'data_ep{episode_idx:02d}_cost{total_cost_value:.4f}.csv'
        csv_path = os.path.join(data_dir, csv_filename)
        df_episode.to_csv(csv_path, index=False)

        # 记录汇总信息
        summary_list.append({
            'Episode': episode_idx,
            'Total_Cost': total_cost_value,
            'Max_Output': max(np.abs(outputs)),  # 粗略的超调/最大响应指标
            'CSV_Path': csv_filename
        })

        # ==========================================
        # 🟡 绘图逻辑 (保持原样，仅修改保存路径)
        # ==========================================
        plt.figure(figsize=(18, 12))
        fig_title = f"Episode {episode_idx} (Total Cost: {total_cost_value:.4f})"
        plt.suptitle(fig_title, fontsize=16)

        plt.subplot(2, 2, 1)
        plt.plot(time_steps, outputs, label='Output', color='C0')
        plt.title('Output Response')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(time_steps, kp_values, label='Kp', color='C1')
        plt.title('$K_p$ Gain')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(time_steps, ki_values, label='Ki', color='C2')
        plt.title('$K_i$ Gain')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(time_steps, kd_values, label='Kd', color='C3')
        plt.title('$K_d$ Gain')
        plt.grid(True)

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig(os.path.join(img_dir, f'plot_ep{episode_idx:02d}.png'))
        plt.close()

    # ==========================================
    # 🟢 关键修改 2：保存汇总表格
    # ==========================================
    # 这对应 5 级要求的“至少3组能说明问题的数据、表格”中的表格部分
    if summary_list:
        df_summary = pd.DataFrame(summary_list)
        # 按 Cost 从小到大排序，最好的模型排在前面
        df_summary = df_summary.sort_values(by='Total_Cost', ascending=True)

        summary_path = os.path.join(data_dir, 'summary_performance_metrics.csv')
        df_summary.to_csv(summary_path, index=False)
        print(f"\n✅ 批处理完成！")
        print(f"1. 源数据已保存至: {data_dir}/ (共 {len(summary_list)} 个CSV文件)")
        print(f"2. 性能汇总表已保存至: {summary_path}")
        print(f"3. 最佳模型是 Episode {df_summary.iloc[0]['Episode']}，Cost 为 {df_summary.iloc[0]['Total_Cost']:.4f}")


if __name__ == "__main__":
    test_ddpg_pid_consolidated()
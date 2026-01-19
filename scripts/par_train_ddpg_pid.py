import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
import torch.multiprocessing as mp

# 导入您项目中的自定义模块
try:
    # 假设 models.py 中定义了 Actor
    from src.agents.models import Actor
    from src.environments.environment_2 import PIDControlEnvironment
    from src.agents.ddpg_agent import DDPGAgent, OUNoise
    from config.config import Config
    from src.utils.utils import set_seed, create_noise_data
    from src.solvers.state_space_2 import StateSpace
except ImportError as e:
    print(f"CRITICAL ERROR: 无法导入必要的模块。请检查依赖文件是否完整。")
    print(f"Import Error: {e}")
    exit()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# --- Worker 函数：完全独立的评估任务 ---
def run_single_evaluation_task(run_id, seed, actor_state_dict):
    """
    运行一个完全独立的回合。 Worker 负责初始化环境、运行、绘图并保存结果。

    Args:
        run_id (int): 任务 ID (0-15).
        seed (int): 唯一的随机种子。
        actor_state_dict (dict): Agent 的 Actor 静态权重。
    """

    # 1. 初始化和配置
    # 注意：每个进程都必须重新设置种子，以保证环境/噪声的随机性
    set_seed(seed)
    config = Config()
    config.DEVICE = 'cpu'

    # 确保保存目录存在
    os.makedirs('results/parallel_test', exist_ok=True)

    # 2. 【关键】为本次运行创建新的随机噪声数据
    tn = config.EPISODE_LENGTH
    noise_data = create_noise_data(tn)
    state_space = StateSpace(config.SYSTEM_CONFIG, noise_data, dt=config.DT, tn=tn)

    env = PIDControlEnvironment(config)
    env.set_state_space(state_space)

    # 3. 创建并加载本地 Actor（仅推理）
    local_actor = Actor(config.STATE_DIM, config.ACTION_DIM, config.HIDDEN_DIM).to(config.DEVICE)
    try:
        local_actor.load_state_dict(actor_state_dict)
        local_actor.eval()
    except Exception as e:
        print(f"[Worker {run_id:02d}] ERROR: Failed to load weights: {e}")
        return (run_id, 0, 0, "Weight Load Error")

    # 4. 运行单个回合
    state = env.reset()
    episode_reward = 0

    for step in range(config.EPISODE_LENGTH):
        # 动作选择：纯策略推理 (无探索噪声)
        state_tensor = torch.FloatTensor(state).to(config.DEVICE).unsqueeze(0)
        action = local_actor(state_tensor).cpu().data.numpy().flatten()
        action = np.clip(action, -1, 1)

        next_state, reward, done, info = env.step(action)

        state = next_state
        episode_reward += reward

        if done:
            break

    # 5. 收集结果和绘图
    y_trajectory = env.state_space.Y.copy()
    metrics = env.get_performance_metrics()

    reward_val = episode_reward[0]
    cost_val = metrics['total_cost'][0]

    # --- 独立绘图并保存 ---
    plt.figure(figsize=(12, 6))
    x_indices = np.arange(len(y_trajectory))

    plt.plot(x_indices, y_trajectory, label='Y 数据值 (电压)', color='blue', linewidth=1.5)
    plt.axhline(0, color='red', linestyle='--', alpha=0.6, label='Set Point')

    plt.title(f"Worker {run_id:02d} | Reward: {reward_val:.2f} | Cost: {cost_val:.2f}", fontsize=16)
    plt.xlabel('时间', fontsize=14)
    plt.ylabel('电压 (Y)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 保存到独立文件
    filename = f'results/parallel_test/run_{run_id:02d}_Y_trajectory.png'
    plt.savefig(filename)
    plt.close()  # 必须关闭图形，防止内存泄漏

    print(f"[Worker {run_id:02d}] 任务完成，Reward: {reward_val:.2f}，图像已保存。")

    # 返回少量信息给主进程
    return (run_id, reward_val, cost_val, filename)


# --- Main 调度函数 ---
def run_parallel_evaluation(num_tasks=16):
    """协调 16 个独立任务的运行。"""

    config = Config()

    print(f"\n--- 易并行评估开始 ({num_tasks} 个独立任务) ---")

    # 1. 准备 Agent 权重（使用一个未训练 Agent 的初始权重）
    # 必须在主进程创建 Agent 以获取可序列化的权重
    try:
        main_agent = DDPGAgent(config)
        # 如果您有训练好的模型，可以在此处加载以进行有意义的评估：
        # main_agent.load_models('models/ddpg_pid_final.pth')
        actor_weights = main_agent.actor.state_dict()
        cpu_weights = {k: v.cpu() for k, v in actor_weights.items()}
    except Exception as e:
        print(f"Main Process Error: 无法初始化 DDPGAgent: {e}")
        return

    # 2. 设置多进程启动方式
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    start_time = time.time()
    all_results = []

    with mp.Pool(processes=num_tasks) as pool:

        # 3. 准备任务参数
        BASE_SEED = 420
        tasks = []
        for i in range(num_tasks):
            worker_seed = BASE_SEED + i
            # 任务结构: (Worker ID, 唯一种子, Actor 权重)
            tasks.append((i, worker_seed, cpu_weights))

        # 4. 运行并行任务
        print(f"Main: 正在分发 {num_tasks} 个独立任务...")
        # starmap 会等待所有任务完成后返回结果列表
        all_results = pool.starmap(run_single_evaluation_task, tasks)
        print(f"Main: 所有 Worker 已返回结果。")

    # 5. 总结结果
    total_time = time.time() - start_time
    print("\n--- 任务总结 ---")
    for run_id, reward, cost, filename in all_results:
        print(f"Run {run_id:02d}: R={reward:.2f}, C={cost:.2f} (File: {filename})")

    print(f"--- 评估完成，总耗时: {total_time:.2f}s ---")


if __name__ == "__main__":
    run_parallel_evaluation(num_tasks=16)
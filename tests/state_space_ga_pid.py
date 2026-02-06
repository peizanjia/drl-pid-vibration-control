import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
import multiprocessing
from functools import partial
import time

# ==========================================
# 1. 环境设置
# ==========================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 自动定位项目根目录 (假设你的目录结构符合之前的设定)
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config.config import Config
    from src.utils.utils import create_noise_data, BeamDisturbanceProjector
    # 注意：不再导入外部 StateSpace，而是直接嵌入逻辑
except ImportError as e:
    print(f"环境导入错误: {e}")
    print("请确保 config.py 和 src/utils/utils.py 存在于正确路径下。")
    sys.exit(1)


# ==========================================
# 2. 核心仿真引擎 (内嵌于评估函数)
# ==========================================
def solve_simulation_core(config_dict, noise_term, tn, dt, kp, ki, kd):
    """
    严格按照用户提供的 StateSpace 逻辑执行仿真。
    不做任何物理层面的修正，仅增加溢出保护。
    """
    # 1. 矩阵组装 (从 config 解析)
    w1, w2 = config_dict['w1'], config_dict['w2']
    z1, z2 = config_dict['z1'], config_dict['z2']
    B1, B2 = config_dict['B1'], config_dict['B2']
    C1, C2 = config_dict['C1'], config_dict['C2']

    A = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [-w1 ** 2, 0, -2 * z1 * w1, 0],
                  [0, -w2 ** 2, 0, -2 * z2 * w2]])
    B = np.array([[0], [0], [B1], [B2]])
    C = np.array([C1, C2, 0, 0])

    # 确保 B 形状
    if B.ndim == 1 or B.shape != (4, 1):
        B = B.reshape(4, 1)

    # 2. 初始化状态
    X = np.zeros((4, tn))
    # Xd = np.zeros((4, tn)) # 如果不需要存储全历史，可以只存当前，节省内存
    Y = np.zeros(tn)

    e = 0.0
    ei = 0.0
    ed = 0.0
    u = 0.0

    F1_arr = noise_term['F1']
    F2_arr = noise_term['F2']

    # 3. 循环求解
    # 增加 Early Stop 标志
    diverged = False

    for i in range(tn - 1):
        # --- 严格保留原代码逻辑开始 ---

        # 0. 状态列向量化
        X_col = X[:, i].reshape(4, 1)

        # 1. 噪声计算
        # 原 compute_noise 逻辑内联
        val1 = F1_arr[i, 1] if i < len(F1_arr) else 0.0
        val2 = F2_arr[i, 1] if i < len(F2_arr) else 0.0
        F1 = np.array([[0], [0], [val1], [val2]])

        val1_next = F1_arr[i + 1, 1] if i + 1 < len(F1_arr) else 0.0
        val2_next = F2_arr[i + 1, 1] if i + 1 < len(F2_arr) else 0.0
        F2 = np.array([[0], [0], [val1_next], [val2_next]])

        # 2. 目标 Xd 计算
        Xd_col = A @ X_col + B * u + F1

        # 3. PID 计算
        e = (C @ X_col).item()
        ei += (e * dt)
        ed = (C @ Xd_col).item()

        # 【用户指定公式】
        u = kp * e - ki * ei - kd * ed

        # 4. RK4 积分
        k1_col = dt * Xd_col
        k2_col = dt * (A @ (X_col + k1_col / 2) + B * u + (F1 + F2) / 2)
        k3_col = dt * (A @ (X_col + k2_col / 2) + B * u + (F1 + F2) / 2)
        k4_col = dt * (A @ (X_col + k3_col) + B * u + F2)

        X_update_col = X_col + (k1_col + 2 * k2_col + 2 * k3_col + k4_col) / 6.0
        X[:, i + 1] = X_update_col.reshape(-1)

        # 5. 输出计算
        Y_next = C @ X_update_col
        Y[i + 1] = Y_next.item()

        # --- 严格保留原代码逻辑结束 ---

        # 6. 安全性检查 (防止数值溢出导致程序崩溃)
        # 如果发散到 1e4 以上，这组参数在物理上已经是废品了，无需继续算到溢出
        if abs(Y[i + 1]) > 1e4:
            diverged = True
            break

    return Y, diverged


# ==========================================
# 3. 并行评估 Worker
# ==========================================
def evaluate_individual_worker(params, config_dict, projector_data, thermal_vector, dt_val, tn_val):
    """
    Worker 函数：评估一组 PID 参数的综合表现
    """
    kp, ki, kd = params

    # 场景定义
    options = ['jitter', 'maneuver', 'impact', 'mixed', 'thermal']
    total_weighted_cost = 0.0

    # 场景权重 (Impact 和 Maneuver 更关键)
    weights = {'jitter': 1.0, 'maneuver': 2.0, 'impact': 3.0, 'mixed': 1.5, 'thermal': 1.0}

    # 蒙特卡洛重复次数 (不惜成本，消除随机误差)
    repeats_per_scenario = 3

    for opt in options:
        mt_data = thermal_vector if opt in ['thermal', 'mixed'] else None

        scenario_costs = []
        for _ in range(repeats_per_scenario):
            try:
                # 动态生成噪声 (不设种子，确保鲁棒性)
                noise_data = create_noise_data(
                    tn=tn_val, dt=dt_val, option=opt,
                    system_config=config_dict,
                    projector_data=projector_data,
                    mt_data=mt_data
                )

                # 调用内嵌仿真器
                Y_trace, diverged = solve_simulation_core(
                    config_dict, noise_data, tn_val, dt_val, kp, ki, kd
                )

                if diverged or np.any(np.isnan(Y_trace)):
                    return 1e15  # 极大惩罚

                # 代价函数: 积分平方误差 (ISE)
                # 越小越好
                cost = np.sum(Y_trace ** 2) * dt_val
                scenario_costs.append(cost)

            except Exception:
                return 1e15  # 捕获任何潜在计算错误

        # 取当前场景下多次重复的平均值 + 标准差惩罚 (奖励稳定性)
        avg_cost = np.mean(scenario_costs)
        std_cost = np.std(scenario_costs)
        total_weighted_cost += (avg_cost + 0.5 * std_cost) * weights[opt]

    return total_weighted_cost


# ==========================================
# 4. 鲁棒遗传算法主类
# ==========================================
class RobustGAPIDOptimizer:
    def __init__(self, conf_obj, bounds, pop_size=500, generations=50, num_cores=8):
        self.conf = conf_obj
        self.config_dict = conf_obj.SYSTEM_CONFIG
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.num_cores = num_cores

        # 准备数据
        self._prepare_thermal_data()
        self._prepare_projector()

        # 初始化种群
        self.population = self._init_population()
        self.best_history = []

    def _prepare_thermal_data(self):
        raw_mt = self.conf.THERMAL_MOMENT
        if isinstance(raw_mt, dict):
            vec = raw_mt.get('M_thermal', np.zeros(self.conf.EPISODE_LENGTH)).flatten()
        else:
            vec = np.array(raw_mt).flatten()

        if len(vec) < self.conf.EPISODE_LENGTH:
            self.thermal_vector = np.pad(vec, (0, self.conf.EPISODE_LENGTH - len(vec)))
        else:
            self.thermal_vector = vec[:self.conf.EPISODE_LENGTH]

    def _prepare_projector(self):
        proj = BeamDisturbanceProjector(L=5.0, n_modes=4)
        self.projector_data = proj.get_static_coeffs()

    def _init_population(self):
        pop = []
        for _ in range(self.pop_size):
            ind = [np.random.uniform(self.bounds[k][0], self.bounds[k][1]) for k in ['kp', 'ki', 'kd']]
            pop.append(ind)
        return pop

    def run(self):
        print("=" * 60)
        print(f"[*] 启动鲁棒 GA 优化器")
        print(f"[*] 核心数: {self.num_cores} | 种群: {self.pop_size} | 代数: {self.generations}")
        print(f"[*] 策略: 5场景混合 + 蒙特卡洛重复验证 + 精英保留")
        print("=" * 60)

        start_time = time.time()

        # 使用 Pool 保持进程常驻
        with multiprocessing.Pool(processes=self.num_cores) as pool:

            for gen in range(self.generations):
                gen_start = time.time()

                # 构造并行任务
                eval_func = partial(
                    evaluate_individual_worker,
                    config_dict=self.config_dict,
                    projector_data=self.projector_data,
                    thermal_vector=self.thermal_vector,
                    dt_val=self.conf.DT,
                    tn_val=self.conf.EPISODE_LENGTH
                )

                # 1. 评估
                fitnesses = np.array(pool.map(eval_func, self.population))

                # 2. 统计
                min_idx = np.argmin(fitnesses)
                current_best_fit = fitnesses[min_idx]
                current_best_ind = self.population[min_idx]

                # 确保全局最优不退化
                if not self.best_history or current_best_fit < self.best_history[-1][0]:
                    global_best_fit = current_best_fit
                    global_best_ind = copy.deepcopy(current_best_ind)
                else:
                    global_best_fit = self.best_history[-1][0]
                    global_best_ind = self.best_history[-1][1]

                self.best_history.append((global_best_fit, global_best_ind))

                valid_count = np.sum(fitnesses < 1e14)

                # 3. 打印日志
                print(f"Gen {gen + 1:03d}/{self.generations} | "
                      f"Cost: {global_best_fit:.4e} | "
                      f"Valid: {valid_count}/{self.pop_size} | "
                      f"PID: [{global_best_ind[0]:.2f}, {global_best_ind[1]:.2f}, {global_best_ind[2]:.2f}] | "
                      f"Time: {time.time() - gen_start:.1f}s")

                # 4. 进化 (选择、交叉、变异)
                self.population = self._evolve(fitnesses, global_best_ind)

        total_time = time.time() - start_time
        print("=" * 60)
        print(f"优化完成. 总耗时: {total_time / 60:.2f} min")
        print(f"最终最优参数: Kp={global_best_ind[0]:.4f}, Ki={global_best_ind[1]:.4f}, Kd={global_best_ind[2]:.4f}")
        return global_best_ind

    def _evolve(self, fitnesses, elite_ind):
        # 精英策略：保留前 5% 或至少前 5 名
        sorted_indices = np.argsort(fitnesses)
        elite_count = max(5, int(self.pop_size * 0.05))
        new_pop = [self.population[i] for i in sorted_indices[:elite_count]]

        # 确保历史最佳一定在种群中
        new_pop[0] = elite_ind

        # 锦标赛选择生成剩余个体
        while len(new_pop) < self.pop_size:
            # 随机选两个
            i1, i2 = np.random.choice(len(self.population), 2, replace=False)
            p1 = self.population[i1]
            p2 = self.population[i2]

            # 强者胜出作为父代1
            parent1 = p1 if fitnesses[i1] < fitnesses[i2] else p2

            i3, i4 = np.random.choice(len(self.population), 2, replace=False)
            parent2 = p3 = self.population[i3] if fitnesses[i3] < fitnesses[i4] else self.population[i4]

            # 交叉 (算术交叉)
            alpha = np.random.rand()
            child = [alpha * g1 + (1 - alpha) * g2 for g1, g2 in zip(parent1, parent2)]

            # 变异 (概率 30%)
            if np.random.rand() < 0.3:
                m_idx = np.random.randint(0, 3)
                # 变异幅度随代数可以衰减，这里简化为固定比例
                scale = (self.bounds['kp'][1] - self.bounds['kp'][0]) * 0.05
                child[m_idx] += np.random.normal(0, scale)

            # 边界约束
            child[0] = np.clip(child[0], self.bounds['kp'][0], self.bounds['kp'][1])
            child[1] = np.clip(child[1], self.bounds['ki'][0], self.bounds['ki'][1])
            child[2] = np.clip(child[2], self.bounds['kd'][0], self.bounds['kd'][1])

            new_pop.append(child)

        return new_pop


# ==========================================
# 5. 结果验证与绘图
# ==========================================
def verify_and_plot(best_pid, conf_obj):
    print("\n[*] 正在生成最优参数验证图...")
    kp, ki, kd = best_pid
    tn = conf_obj.EPISODE_LENGTH
    dt = conf_obj.DT

    # 1. 生成一个典型的恶劣工况 (Impact) 进行展示
    proj = BeamDisturbanceProjector()
    coeffs = proj.get_static_coeffs()
    noise = create_noise_data(tn, dt, 'impact', conf_obj.SYSTEM_CONFIG, projector_data=coeffs)

    # 2. 运行仿真
    Y, _ = solve_simulation_core(conf_obj.SYSTEM_CONFIG, noise, tn, dt, kp, ki, kd)

    # 3. 绘图
    t = np.arange(tn) * dt

    plt.figure(figsize=(12, 6))
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 12

    # 时域响应
    plt.subplot(1, 2, 1)
    plt.plot(t, Y, 'b-', linewidth=1.5, label='Response (Y)')
    plt.title(f'Optimized Response (Impact Scenario)\nKp={kp:.1f}, Ki={ki:.1f}, Kd={kd:.1f}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    # 局部放大 (看震荡衰减)
    plt.subplot(1, 2, 2)
    zoom_idx = int(tn * 0.6)  # 看后半段
    plt.plot(t[zoom_idx:], Y[zoom_idx:], 'r-', linewidth=1.5, label='Tail Damping')
    plt.title('Damping Performance (Tail)')
    plt.xlabel('Time (s)')
    plt.legend()

    plt.tight_layout()
    plt.show()


# ==========================================
# 6. 程序入口
# ==========================================
if __name__ == "__main__":
    multiprocessing.freeze_support()

    # 1. 加载配置
    config_obj = Config()

    # 2. 设置宽泛但合理的边界
    # 提示：Kd 对振动抑制非常关键，上限设高一点
    bounds = {
        'kp': [0.0, 500.0],
        'ki': [0.0, 50.0],
        'kd': [0.0, 200.0]
    }

    # 3. 实例化优化器 (不惜计算成本配置)
    optimizer = RobustGAPIDOptimizer(
        conf_obj=config_obj,
        bounds=bounds,
        pop_size=1000,  # 种群大小
        generations=300,  # 代数
        num_cores=14  # 核心数
    )

    # 4. 运行
    best_params = optimizer.run()

    # 5. 验证绘图
    verify_and_plot(best_params, config_obj)
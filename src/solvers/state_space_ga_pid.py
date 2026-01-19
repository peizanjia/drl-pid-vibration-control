import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import os
import multiprocessing
from functools import partial
from config.config import Config
from src.utils.utils import set_seed, create_noise_data

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ==========================================
# 1. 全局配置与辅助
# ==========================================
tn_sim = 10000
dt_sim = 0.01


# ==========================================
# 2. StateSpace 类 (保持不变)
# ==========================================
class StateSpace:
    def __init__(self, config, noise_term, dt=0.01, tn=1000, dx=0.005, kp=0.0, ki=0.0, kd=0.0):
        # 这里的 config 预期是一个字典
        self.config = config
        self.noise_term = noise_term
        self.dt = dt
        self.tn = tn
        self.dx = dx
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.e = 0
        self.ei = 0
        self.ed = 0

        self.X = np.zeros((4, tn))
        self.Xd = np.zeros((4, tn))
        self.Y = np.zeros(tn)
        self.u = 0

        self.F1 = noise_term['F1']
        self.F2 = noise_term['F2']

        self.A, self.B, self.C = self.assemble_mat()
        self.external_controller_callback = None

        self.kp_history = np.zeros(tn)
        self.ki_history = np.zeros(tn)
        self.kd_history = np.zeros(tn)

    def set_external_controller(self, controller_callback):
        self.external_controller_callback = controller_callback

    def assemble_mat(self):
        # 确保这里通过字典键值访问
        w1, w2 = self.config['w1'], self.config['w2']
        z1, z2 = self.config['z1'], self.config['z2']
        B1, B2 = self.config['B1'], self.config['B2']
        C1, C2 = self.config['C1'], self.config['C2']

        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [-w1 ** 2, 0, -2 * z1 * w1, 0],
                      [0, -w2 ** 2, 0, -2 * z2 * w2]])
        B = np.array([[0], [0], [B1], [B2]])
        C = np.array([C1, C2, 0, 0])
        return A, B, C

    def __compute_noise(self, idx):
        if idx < self.tn:
            val1 = self.F1[idx, 1] if idx < len(self.F1) else 0
            val2 = self.F2[idx, 1] if idx < len(self.F2) else 0
            F = np.array([[0], [0], [val1], [val2]])
        else:
            F = np.zeros((4, 1))
        return F

    def solve(self):
        if self.B.ndim == 1 or self.B.shape != (4, 1):
            self.B = self.B.reshape(4, 1)

        for i in range(self.tn - 1):
            self.kp_history[i] = self.kp
            self.ki_history[i] = self.ki
            self.kd_history[i] = self.kd

            X_col = self.X[:, i].reshape(4, 1)
            F1 = self.__compute_noise(i)
            F2 = self.__compute_noise(i + 1)

            if F1.ndim == 1: F1 = F1.reshape(4, 1)
            if F2.ndim == 1: F2 = F2.reshape(4, 1)

            Xd_col = self.A @ X_col + self.B * self.u + F1
            self.Xd[:, i] = Xd_col.reshape(-1)

            self.e = (self.C @ X_col).item()
            self.ei += (self.e * self.dt)
            self.ed = (self.C @ Xd_col).item()

            self.u = self.kp * self.e + self.ki * self.ei + self.kd * self.ed

            if self.external_controller_callback is not None:
                new_kp, new_ki, new_kd = self.external_controller_callback(self.Y[i], i * self.dt)
                self.kp, self.ki, self.kd = new_kp, new_ki, new_kd

            k1_col = self.dt * Xd_col
            k2_col = self.dt * (self.A @ (X_col + k1_col / 2) + self.B * self.u + (F1 + F2) / 2)
            k3_col = self.dt * (self.A @ (X_col + k2_col / 2) + self.B * self.u + (F1 + F2) / 2)
            k4_col = self.dt * (self.A @ (X_col + k3_col) + self.B * self.u + F2)

            X_update_col = X_col + (k1_col + 2 * k2_col + 2 * k3_col + k4_col) / 6.0

            if np.any(np.abs(X_update_col) > 1e10):
                self.X[:, i + 1:] = np.nan
                self.Y[i + 1:] = np.nan
                break

            self.X[:, i + 1] = X_update_col.reshape(-1)
            self.Y[i + 1] = (self.C @ X_update_col).item()

        self.kp_history[-1] = self.kp
        self.ki_history[-1] = self.ki
        self.kd_history[-1] = self.kd


# ==========================================
# 3. 独立的并行工作函数 (必须定义在类外面)
# ==========================================
def evaluate_individual(params, config_dict, noise_data, w_y, w_yd, dt_val, tn_val):
    """
    这个函数将被复制到每个 CPU 核心上独立运行。
    它不依赖于 GAPIDOptimizer 实例，只依赖传入的数据。
    """
    kp, ki, kd = params

    # 实例化仿真
    sim = StateSpace(
        config=config_dict,
        noise_term=noise_data,
        tn=tn_val,
        dt=dt_val,
        kp=kp, ki=ki, kd=kd
    )

    try:
        sim.solve()

        # 结果提取
        Y = sim.Y
        Xd = sim.Xd
        Y_dot = sim.C @ Xd

        # --- 发散检测 ---
        if np.any(np.isnan(Y)) or np.any(np.isinf(Y)) or np.max(np.abs(Y)) > 1e5:
            return 1e12  # 发散惩罚

        # --- 代价计算 (归一化为积分形式) ---
        # 乘以 dt 使其物理意义明确 (积分)
        loss = np.sum(w_y * (Y ** 2) + w_yd * (Y_dot ** 2)) * dt_val

        return loss

    except Exception:
        return 1e12


# ==========================================
# 4. 遗传算法优化器 (并行版)
# ==========================================

class GAPIDOptimizerParallel:
    def __init__(self, config, noise, bounds, pop_size=1000, generations=500, mutation_rate=0.3, num_cores=16):
        self.config = config
        self.noise = noise
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.num_cores = num_cores  # 核心数

        self.w_y = 1.0
        self.w_yd = 10.0

        self.best_fitness_history = []
        self.best_params_history = []
        self.valid_inds_history = []  # 新增：记录每一代存活数量

        self.population = self.init_population()

    def init_population(self):
        pop = []
        for _ in range(self.pop_size):
            kp = np.random.uniform(self.bounds['kp'][0], self.bounds['kp'][1])
            ki = np.random.uniform(self.bounds['ki'][0], self.bounds['ki'][1])
            kd = np.random.uniform(self.bounds['kd'][0], self.bounds['kd'][1])
            pop.append([kp, ki, kd])
        return pop

    # 注意：原 calculate_cost 方法已被移除，逻辑移动到了全局函数 evaluate_individual

    def select(self, population, fitnesses):
        selected = []
        # 精英保留
        sorted_indices = np.argsort(fitnesses)
        best_idx = sorted_indices[0]
        selected.append(population[best_idx])

        # 锦标赛选择
        # 预先生成随机索引以加速
        indices = np.arange(len(population))
        for _ in range(self.pop_size - 1):
            i1, i2 = np.random.choice(indices, 2, replace=False)
            if fitnesses[i1] < fitnesses[i2]:
                selected.append(population[i1])
            else:
                selected.append(population[i2])
        return selected

    def crossover(self, parent1, parent2):
        alpha = np.random.rand()
        child1 = [alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)]
        child2 = [(1 - alpha) * p1 + alpha * p2 for p1, p2 in zip(parent1, parent2)]
        return child1, child2

    def mutate(self, individual):
        new_ind = list(individual)  # Copy
        for i in range(3):
            if np.random.rand() < self.mutation_rate:
                keys = ['kp', 'ki', 'kd']
                key = keys[i]
                span = self.bounds[key][1] - self.bounds[key][0]
                sigma = span * 0.1
                noise = np.random.normal(0, sigma)
                new_ind[i] += noise
                new_ind[i] = np.clip(new_ind[i], self.bounds[key][0], self.bounds[key][1])
        return new_ind

    def run(self):
        print(f"开始并行遗传算法优化 (Cores: {self.num_cores})...")
        print(f"种群大小: {self.pop_size}, 代数: {self.generations}")
        print("-" * 60)

        start_time = time.time()

        # 初始化进程池
        # 我们在这里创建 Pool，这样可以复用，不必每一代都重新创建销毁
        with multiprocessing.Pool(processes=self.num_cores) as pool:

            for gen in range(self.generations):
                gen_start = time.time()

                # --- 并行计算部分 ---
                # 使用 partial 固定住 config, noise 等不变的参数
                # 这样 map 只需要分发 population (变动的参数)
                eval_func = partial(
                    evaluate_individual,
                    config_dict=self.config,
                    noise_data=self.noise,
                    w_y=self.w_y,
                    w_yd=self.w_yd,
                    dt_val=dt_sim,
                    tn_val=tn_sim
                )

                # pool.map 会自动将 self.population 中的每个 individual 传给 eval_func
                # 并返回结果列表，顺序与 population 一致
                fitnesses = pool.map(eval_func, self.population)
                fitnesses = np.array(fitnesses)

                # --- 统计与记录 ---
                valid_count = np.sum(fitnesses < 1e10)
                self.valid_inds_history.append(valid_count)

                best_idx = np.argmin(fitnesses)
                best_cost = fitnesses[best_idx]
                best_ind = self.population[best_idx]

                self.best_fitness_history.append(best_cost)
                self.best_params_history.append(best_ind)

                # 计算耗时
                gen_time = time.time() - gen_start

                # 动态打印 (每10代或者刚开始时打印)
                if gen % 10 == 0 or gen == 0:
                    print(f"Gen [{gen + 1}/{self.generations}] | "
                          f"Time: {gen_time:.2f}s | "
                          f"Best Cost: {best_cost:.4e} | "
                          f"Valid: {valid_count} | "
                          f"PID: {best_ind[0]:.1f}, {best_ind[1]:.1f}, {best_ind[2]:.1f}")

                # --- 进化操作 (串行，极快) ---
                selected_pop = self.select(self.population, fitnesses)

                next_pop = []
                next_pop.append(selected_pop[0])  # Elite

                idx = 1
                while len(next_pop) < self.pop_size:
                    if idx + 1 < len(selected_pop):
                        p1 = selected_pop[idx]
                        p2 = selected_pop[idx + 1]
                        c1, c2 = self.crossover(p1, p2)
                        next_pop.append(self.mutate(c1))
                        if len(next_pop) < self.pop_size:
                            next_pop.append(self.mutate(c2))
                        idx += 2
                    else:
                        next_pop.append(self.mutate(selected_pop[idx]))
                        idx += 1

                self.population = next_pop

        total_time = time.time() - start_time
        print("-" * 60)
        print(f"优化完成. 总耗时: {total_time:.2f}s")
        print(f"平均每代耗时: {total_time / self.generations:.2f}s")
        print(f"最终最优 Cost: {self.best_fitness_history[-1]:.4e}")
        return self.best_params_history[-1]


# ==========================================
# 5. 主程序入口
# ==========================================

if __name__ == "__main__":
    # 多进程必须在 __main__ 保护下运行 (特别是 Windows/MacOS)
    multiprocessing.freeze_support()

    set_seed(42)

    # 调整了 bounds，给 Kd 更多空间
    pid_bounds = {
        'kp': [100, 200],
        'ki': [0, 1],
        'kd': [0, 1]  # 增加 Kd 上界以匹配 Kp
    }

    config = Config()
    tn = config.EPISODE_LENGTH
    # 注意：noise_data 比较大，传递给子进程会有一定开销
    # 但相比于 solve 的计算量，这点开销是值得的
    noise_data = create_noise_data(tn)

    # 实例化并行优化器，请求 16 核
    ga = GAPIDOptimizerParallel(
        config.SYSTEM_CONFIG,
        noise_data,
        pid_bounds,
        pop_size=500,
        generations=50,  # 可以适当减少代数，因为种群大且并行快
        num_cores=16  # 指定核心数
    )

    # 运行
    best_pid = ga.run()

    # ==========================================
    # 验证与绘图
    # ==========================================
    final_sim = StateSpace(
        config,  # 这里传入原始 Config 对象给最后一次单次仿真
        noise_data,
        tn=tn_sim,
        dt=dt_sim,
        kp=best_pid[0], ki=best_pid[1], kd=best_pid[2]
    )
    final_sim.solve()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Cost 历史
    axs[0, 0].plot(ga.best_fitness_history, 'r-', linewidth=2)
    axs[0, 0].set_title('Fitness Convergence')
    axs[0, 0].set_xlabel('Generation')
    axs[0, 0].set_ylabel('Cost')
    axs[0, 0].grid(True)

    # 2. 存活个体历史 (Valid Inds) - 新增图表
    axs[0, 1].plot(ga.valid_inds_history, 'g-', linewidth=2)
    axs[0, 1].set_title('Population Stability (Valid Individuals)')
    axs[0, 1].set_xlabel('Generation')
    axs[0, 1].set_ylabel('Count (Max 1000)')
    axs[0, 1].grid(True)

    # 3. 最优响应
    time_axis = np.arange(tn_sim) * dt_sim
    axs[1, 0].plot(time_axis, final_sim.Y, 'b-', label='Optimized Y')
    axs[1, 0].set_title(f'Optimized Response\nKp={best_pid[0]:.1f}, Ki={best_pid[1]:.1f}, Kd={best_pid[2]:.1f}')
    axs[1, 0].grid(True)

    # 4. 状态 X
    axs[1, 1].plot(time_axis, final_sim.X[0, :], label='X[0]')
    axs[1, 1].plot(time_axis, final_sim.X[2, :], label='X[2]')
    axs[1, 1].set_title('Internal States')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()
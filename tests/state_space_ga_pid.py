import numpy as np
import copy
import os
import multiprocessing
from functools import partial
from config.config import Config
from src.utils.utils import create_noise_data

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

            self.u = self.kp * self.e - self.ki * self.ei - self.kd * self.ed

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
def evaluate_individual_robust(params, config_dict, projector_data, w_y, w_yd, dt_val, tn_val):
    """
    鲁棒性评估：在 5 种不同的扰动环境下评估同一组 PID 参数
    """
    kp, ki, kd = params
    options = ['jitter', 'maneuver', 'impact', 'mixed', 'thermal']
    total_loss = 0.0

    # 模拟热变形需要的简化 mt_data (如果没有真实数据，Worker 内部构造)
    # 假设热变形是一个缓慢变化的偏置
    mt_data_dummy = np.sin(np.linspace(0, np.pi, tn_val)) * 0.05

    for opt in options:
        try:
            # 1. 动态生成噪声 (不设种子，确保随机性)
            noise_data = create_noise_data(
                tn=tn_val,
                dt=dt_val,
                option=opt,
                system_config=config_dict,
                projector_data=projector_data,
                mt_data=mt_data_dummy
            )

            # 2. 实例化仿真
            sim = StateSpace(
                config=config_dict,
                noise_term=noise_data,
                tn=tn_val,
                dt=dt_val,
                kp=kp, ki=ki, kd=kd
            )

            sim.solve()

            # 3. 结果提取与发散检测
            Y = sim.Y
            if np.any(np.isnan(Y)) or np.max(np.abs(Y)) > 1e4:
                return 1e15  # 只要有一种模式发散，该个体即被淘汰

            Y_dot = sim.C @ sim.Xd
            loss = np.sum(w_y * (Y ** 2) + w_yd * (Y_dot ** 2)) * dt_val
            total_loss += loss

        except Exception:
            return 1e15

    return total_loss / len(options)  # 返回平均代价


# ==========================================
# 4. 遗传算法优化器 (并行版)
# ==========================================

class GAPIDOptimizerParallel:
    def __init__(self, config_dict, projector_data, bounds, pop_size=200, generations=50, mutation_rate=0.3,
                 num_cores=16):
        self.config_dict = config_dict
        self.projector_data = projector_data  # 传递投影系数字典
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.num_cores = num_cores

        self.w_y = 1.0
        self.w_yd = 10.0  # 增加对速度项的惩罚，有利于抑制振动

        self.best_fitness_history = []
        self.best_params_history = []
        self.population = self.init_population()

    def init_population(self):
        pop = []
        for _ in range(self.pop_size):
            ind = [np.random.uniform(self.bounds[k][0], self.bounds[k][1]) for k in ['kp', 'ki', 'kd']]
            pop.append(ind)
        return pop

    def select(self, population, fitnesses):
        # 严格的精英保留：找到当前代绝对最优
        sorted_indices = np.argsort(fitnesses)
        best_individual = copy.deepcopy(population[sorted_indices[0]])

        selected = [best_individual]  # 保留精英

        # 锦标赛选择剩余个体
        indices = np.arange(len(population))
        for _ in range(self.pop_size - 1):
            i1, i2 = np.random.choice(indices, 2, replace=False)
            selected.append(copy.deepcopy(population[i1] if fitnesses[i1] < fitnesses[i2] else population[i2]))
        return selected

    def crossover(self, parent1, parent2):
        # 算术交叉
        alpha = np.random.rand()
        child = [alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)]
        return child

    def mutate(self, individual):
        for i, key in enumerate(['kp', 'ki', 'kd']):
            if np.random.rand() < self.mutation_rate:
                span = self.bounds[key][1] - self.bounds[key][0]
                individual[i] += np.random.normal(0, span * 0.05)
                individual[i] = np.clip(individual[i], self.bounds[key][0], self.bounds[key][1])
        return individual

    def run(self):
        print(f"开始鲁棒性优化 (模式: 5种环境混合) | 核心数: {self.num_cores}")

        with multiprocessing.Pool(processes=self.num_cores) as pool:
            for gen in range(self.generations):
                # 包装评估函数
                eval_func = partial(
                    evaluate_individual_robust,
                    config_dict=self.config_dict,
                    projector_data=self.projector_data,
                    w_y=self.w_y,
                    w_yd=self.w_yd,
                    dt_val=dt_sim,
                    tn_val=tn_sim
                )

                fitnesses = np.array(pool.map(eval_func, self.population))

                best_idx = np.argmin(fitnesses)
                self.best_fitness_history.append(fitnesses[best_idx])
                self.best_params_history.append(copy.deepcopy(self.population[best_idx]))

                if gen % 5 == 0:
                    print(f"Gen {gen:03d} | Best Cost: {fitnesses[best_idx]:.4e} | PID: {self.population[best_idx]}")

                # 进化操作
                selected_pop = self.select(self.population, fitnesses)
                next_pop = [selected_pop[0]]  # 确保存放精英

                while len(next_pop) < self.pop_size:
                    p1, p2 = np.random.choice(len(selected_pop), 2, replace=False)
                    child = self.crossover(selected_pop[p1], selected_pop[p2])
                    next_pop.append(self.mutate(child))

                self.population = next_pop

        return self.best_params_history[-1]


# ==========================================
# 5. 主程序入口
# ==========================================

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # 1. 初始化物理投影器 (提取静态系数用于并行)
    from src.utils.utils import BeamDisturbanceProjector

    projector = BeamDisturbanceProjector(L=5.0, n_modes=4)
    proj_data = projector.get_static_coeffs()

    # 2. 配置
    config = Config()
    system_dict = config.SYSTEM_CONFIG

    pid_bounds = {
        'kp': [0, 200],
        'ki': [0, 50],
        'kd': [0, 30]  # 动力学系统中 Kd 对阻尼贡献极大
    }

    # 3. 运行优化
    # 注意：不要在此处 set_seed，让 Worker 内部的随机数自然发挥
    ga = GAPIDOptimizerParallel(
        config_dict=system_dict,
        projector_data=proj_data,
        bounds=pid_bounds,
        pop_size=1000,
        generations=200,
        num_cores=14
    )

    best_pid = ga.run()
    print(f"最优参数确认为: {best_pid}")
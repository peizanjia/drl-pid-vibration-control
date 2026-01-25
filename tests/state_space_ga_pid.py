import numpy as np
import copy
import os
import sys
import multiprocessing
from functools import partial

# ==========================================
# 1. 环境与路径补丁
# ==========================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 自动定位项目根目录
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import Config
from src.utils.utils import create_noise_data, BeamDisturbanceProjector
from src.solvers.state_space_2 import StateSpace  # 确保你的 StateSpace 类支持 solve() 方法


# ==========================================
# 2. 独立的并行评估函数 (必须在类外)
# ==========================================
def evaluate_individual_robust(params, config_dict, projector_data, thermal_vector, dt_val, tn_val):
    """
    在 5 种工况下评估同一组 PID 参数的鲁棒性
    thermal_vector: 外部导入并对齐后的热载荷数组
    """
    kp, ki, kd = params
    # 场景列表
    options = ['jitter', 'maneuver', 'impact', 'mixed', 'thermal']
    total_loss = 0.0

    # 权重系数
    w_y = 1.0
    w_yd = 10.0  # 对速度(振动)的惩罚更重

    for opt in options:
        try:
            # 根据场景准备热数据
            mt_data = thermal_vector if opt in ['thermal', 'mixed'] else None

            # 1. 生成当前场景的噪声
            # 注意：此处不设固定种子，让 GA 在轻微扰动中筛选最稳健的参数
            noise_data = create_noise_data(
                tn=tn_val,
                dt=dt_val,
                option=opt,
                system_config=config_dict,
                projector_data=projector_data,
                mt_data=mt_data
            )

            # 2. 仿真
            sim = StateSpace(
                config=config_dict,
                noise_term=noise_data,
                tn=tn_val,
                dt=dt_val,
                kp=kp, ki=ki, kd=kd
            )
            sim.solve()

            # 3. 计算 Cost (能量泛函)
            y = sim.Y
            if np.any(np.isnan(y)) or np.max(np.abs(y)) > 0.5:  # 发散判据
                return 1e10

            # 速度项 (从 Xd 中提取或近似)
            # 假设 StateSpace 记录了输出 Y 的导数，或者通过 C @ Xd 计算
            y_dot = sim.C @ sim.Xd

            # 计算积分损失
            loss = np.sum(w_y * (y ** 2) + w_yd * (y_dot ** 2)) * dt_val
            total_loss += loss

        except Exception:
            return 1e10

    return total_loss / len(options)


# ==========================================
# 3. GA 优化器类
# ==========================================
class GAPIDOptimizerParallel:
    def __init__(self, conf, bounds, pop_size=100, generations=50, num_cores=8):
        self.conf = conf
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.num_cores = num_cores

        # --- 外部热载荷预处理 ---
        raw_mt = conf.THERMAL_MOMENT
        if isinstance(raw_mt, dict):
            self.thermal_vector = raw_mt.get('M_thermal', np.zeros(conf.EPISODE_LENGTH)).flatten()
        else:
            self.thermal_vector = np.array(raw_mt).flatten()

        # 强制对齐长度
        if len(self.thermal_vector) < conf.EPISODE_LENGTH:
            self.thermal_vector = np.pad(self.thermal_vector, (0, conf.EPISODE_LENGTH - len(self.thermal_vector)))
        else:
            self.thermal_vector = self.thermal_vector[:conf.EPISODE_LENGTH]

        # 预计算投影系数
        proj = BeamDisturbanceProjector(L=5.0, n_modes=4)
        self.projector_data = proj.get_static_coeffs()

        self.population = self.init_population()

    def init_population(self):
        pop = []
        for _ in range(self.pop_size):
            ind = [np.random.uniform(self.bounds[k][0], self.bounds[k][1]) for k in ['kp', 'ki', 'kd']]
            pop.append(ind)
        return pop

    def run(self):
        print(f"[*] 启动鲁棒性 GA 优化 | 核心数: {self.num_cores} | 种群: {self.pop_size}")
        print(f"[*] 外部热载荷已对齐，长度: {len(self.thermal_vector)}")

        best_ind = None
        best_fit = float('inf')

        with multiprocessing.Pool(processes=self.num_cores) as pool:
            for gen in range(self.generations):
                # 包装评估函数，注入外部 Thermal 数组
                eval_func = partial(
                    evaluate_individual_robust,
                    config_dict=self.conf.SYSTEM_CONFIG,
                    projector_data=self.projector_data,
                    thermal_vector=self.thermal_vector,
                    dt_val=self.conf.DT,
                    tn_val=self.conf.EPISODE_LENGTH
                )

                # 并行计算
                fitnesses = pool.map(eval_func, self.population)

                # 找到当前代最优
                min_idx = np.argmin(fitnesses)
                if fitnesses[min_idx] < best_fit:
                    best_fit = fitnesses[min_idx]
                    best_ind = copy.deepcopy(self.population[min_idx])

                print(f"Gen {gen:03d} | Best Cost: {best_fit:.6e} | Best PID: {best_ind}")

                # --- 进化操作 (选择、交叉、变异) ---
                self.population = self.evolve(fitnesses)

        return best_ind

    def evolve(self, fitnesses):
        # 简单的精英选择 + 交叉变异
        sorted_indices = np.argsort(fitnesses)
        new_pop = [copy.deepcopy(self.population[i]) for i in sorted_indices[:2]]  # 保留前2名精英

        while len(new_pop) < self.pop_size:
            # 锦标赛选择
            idx1, idx2 = np.random.choice(range(self.pop_size), 2, replace=False)
            parent1 = self.population[idx1] if fitnesses[idx1] < fitnesses[idx2] else self.population[idx2]

            idx3, idx4 = np.random.choice(range(self.pop_size), 2, replace=False)
            parent2 = self.population[idx3] if fitnesses[idx3] < fitnesses[idx4] else self.population[idx4]

            # 交叉
            child = [(p1 + p2) / 2 + np.random.normal(0, 1.0) for p1, p2 in zip(parent1, parent2)]

            # 变异与边界限制
            for i, key in enumerate(['kp', 'ki', 'kd']):
                child[i] = np.clip(child[i], self.bounds[key][0], self.bounds[key][1])

            new_pop.append(child)
        return new_pop


# ==========================================
# 4. 主程序执行
# ==========================================
if __name__ == "__main__":
    # 多进程必须保护
    multiprocessing.freeze_support()

    # 1. 加载配置
    conf_obj = Config()

    # 2. 设定搜索边界
    # 注意：对于带 Thermal 的系统，Ki 不能为 0，否则无法消除静差
    bounds = {
        'kp': [20.0, 200.0],
        'ki': [1.0, 50.0],
        'kd': [5.0, 40.0]
    }

    # 3. 运行优化
    optimizer = GAPIDOptimizerParallel(
        conf=conf_obj,
        bounds=bounds,
        pop_size=200,  # 根据你的 CPU 核心数调整
        generations=100,
        num_cores=14  # 你的机器有14核可用
    )

    best_pid = optimizer.run()
    print(f"\n[FINAL] 鲁棒优化后的最优 PID 参数: {best_pid}")
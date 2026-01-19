import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
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
        if np.any(np.isnan(Y)) or np.any(np.isinf(Y)) or np.max(np.abs(Y)) > 1e2:
            return np.nan  # 发散惩罚

        # --- 代价计算 (归一化为积分形式) ---
        # 乘以 dt 使其物理意义明确 (积分)
        loss = np.sum(w_y * (Y ** 2) + w_yd * (Y_dot ** 2)) * dt_val

        return loss

    except Exception:
        return np.nan


def compute_loss(i, config_system, noise_data, w_yd, dt, tn):
    """封装单次计算任务"""
    kp = i + 100
    params = (kp, 0, 0)
    # 调用原有的评估函数
    return evaluate_individual(params, config_system, noise_data, 1, w_yd, dt, tn)

# ==========================================
# 5. 主程序入口
# ==========================================

if __name__ == "__main__":
    multiprocessing.freeze_support()

    config = Config()
    tn = config.EPISODE_LENGTH


    while True:
        try:
            w_yd_input = input("input w_yd (or 'q' to quit): ")
            if w_yd_input.lower() == 'q': break
            w_yd = float(w_yd_input)
        except ValueError:
            continue

        # --- 并行计算开始 ---
        # 准备参数列表：每个元素都是 compute_loss 的参数元组
        noise_data = create_noise_data(tn)
        tasks = [
            (i, config.SYSTEM_CONFIG, noise_data, w_yd, config.DT, tn)
            for i in range(100)
        ]

        # 开启 16 个进程
        with multiprocessing.Pool(processes=16) as pool:
            # starmap 可以自动解包 tasks 中的元组并传给 compute_loss
            results = pool.starmap(compute_loss, tasks)

        loss_value = np.array(results)
        # --- 并行计算结束 ---

        min_idx = np.nanargmin(loss_value)
        kp = min_idx + 100

        print(f"Optimal Kp index: {min_idx}, Value: {100 + min_idx}, Min Loss: {loss_value[min_idx]}")
        opt_ss = StateSpace(config.SYSTEM_CONFIG, noise_data, tn=tn, dx=0.005, kp=kp)
        opt_ss.solve()
        plt.plot(opt_ss.Y)
        plt.show()

        plt.plot(loss_value)
        plt.axvline(min_idx, color='r', linestyle='--', label='Min Loss')
        plt.legend()
        plt.show()
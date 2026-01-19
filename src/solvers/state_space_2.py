import numpy as np
import matplotlib.pyplot as plt
import os
from config.config import Config
from config.config_loader import load_mat
from src.utils.utils import set_seed, create_noise_data

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# X'(t) = AX(t) + Bu(t) + F(t)
# Y(t) = CX(t)

class StateSpace:

    def __init__(self, config, noise_term, dt=0.01, tn=10000, dx=0.005, kp=0.0, ki=0.0, kd=0.0):
        self.config = config
        self.noise_term = noise_term
        self.dt = dt
        self.tn = tn
        self.dx = dx

        # PID参数 - 初始值
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # PID状态变量
        self.e = 0
        self.ei = 0
        self.ed = 0

        # 状态和输出数组
        self.X = np.zeros((4, tn))
        self.Xd = np.zeros((4, tn))
        self.Y = np.zeros(tn)
        self.u = 0

        # --- 卡尔曼滤波器初始化 ---
        self.X_hat = np.zeros((4, tn))
        self.Xd_hat = np.zeros((4, tn))

        # 估计协方差 P
        self.P = np.eye(4) * 0.1
        # 记录 P 的对角线元素 (方差)，用于绘制 3-sigma 包络线
        # 形状: [4, tn], 对应 4 个状态
        self.P_diag_history = np.zeros((4, tn))

        # 过程噪声协方差 Q
        self.Q = np.eye(4) * 1e-5

        # 观测噪声协方差 R
        self.R_val = 1e-3

        # 噪声项
        self.F1 = self.noise_term['F1']
        self.F2 = self.noise_term['F2']

        # 系统矩阵
        self.A, self.B, self.C = self.assemble_mat()
        self.Ad = np.eye(4) + self.A * self.dt
        self.Bd = self.B * self.dt

        # 外部控制器回调函数 - 初始为None，需要在solve前设置
        self.external_controller_callback = None

        # 存储每个时间步的PID参数用于分析
        self.kp_history = np.zeros(tn)
        self.ki_history = np.zeros(tn)
        self.kd_history = np.zeros(tn)

    def set_external_controller(self, controller_callback):
        """
        设置外部控制器回调函数

        Args:
            controller_callback: 函数，接受(Y, time)作为输入，返回(kp, ki, kd)
        """
        self.external_controller_callback = controller_callback

    def assemble_mat(self):
        w1 = self.config['w1']
        w2 = self.config['w2']
        z1 = self.config['z1']
        z2 = self.config['z2']
        B1 = self.config['B1']
        B2 = self.config['B2']
        C1 = self.config['C1']
        C2 = self.config['C2']

        A = np.array([[0,        0,        1,            0           ],
                      [0,        0,        0,            1           ],
                      [-w1 ** 2, 0,        -2 * z1 * w1, 0           ],
                      [0,        -w2 ** 2, 0,            -2 * z2 * w2]])

        B = np.array([[0], [0], [B1], [B2]])

        C = np.array([C1, C2, 0, 0])

        return A, B, C

    def __compute_noise(self, idx):
        if idx < self.tn:
            F = np.array(
                [[0], [0], [self.F1[idx, 1]], [self.F2[idx, 1]]])
        else:
            F = np.zeros((4, 1))
        return F

    def solve(self):
        # 确保矩阵形状正确
        if self.B.ndim == 1 or self.B.shape != (4, 1):
            self.B = self.B.reshape(4, 1)

        C_mat = self.C.reshape(1, 4)
        R_mat = np.array([[self.R_val]])

        # 记录初始的 P
        self.P_diag_history[:, 0] = np.diag(self.P)

        # 检查是否设置了外部控制器
        if self.external_controller_callback is None:
            print("警告: 未设置外部控制器，将使用固定PID参数")

        for i in range(self.tn - 1):
            # 存储当前PID参数
            self.kp_history[i] = self.kp
            self.ki_history[i] = self.ki
            self.kd_history[i] = self.kd

            # 0. 关键：将当前状态 X[:, i] 转换为 (4, 1) 列向量
            X_col = self.X[:, i].reshape(4, 1)  # 转换为 (4, 1)
            X_hat_col = self.X_hat[:, i].reshape(4, 1)

            # --- 1. 噪声计算与修正 ---
            F1 = self.__compute_noise(i)
            F2 = self.__compute_noise(i + 1)

            # 确保 F1 和 F2 是 (4, 1)
            if F1.ndim == 1 or F1.shape != (4, 1):
                F1 = F1.reshape(4, 1)
            if F2.ndim == 1 or F2.shape != (4, 1):
                F2 = F2.reshape(4, 1)

            # --- 2. 目标 Xd 赋值 ---
            Xd_col = self.A @ X_col + self.B * self.u + F1
            Xd_hat_col = self.A @ X_hat_col + self.B * self.u + F1
            self.Xd[:, i] = Xd_col.reshape(-1)
            self.Xd_hat[:, i] = Xd_hat_col.reshape(-1)

            # --- 3. PID 计算 ---
            # self.e = -2 * X_col[0] + 2 * X_col[1]
            # self.ei += (-2 * X_col[0] + 2 * X_col[1]) * self.dt
            # self.ed = -2 * X_col[2] + 2 * X_col[3]
            self.e = (self.C @ X_col)
            self.ei += (self.e * self.dt)
            self.ed = (self.C @ Xd_col)

            self.u = self.kp * self.e - self.ki * self.ei - self.kd * self.ed  # u 是标量

            # --- 4. 与外部神经网络控制器交互 ---
            if self.external_controller_callback is not None:
                current_time = i * self.dt
                current_output = self.Y[i]

                # 调用外部控制器获取新的PID参数
                new_kp, new_ki, new_kd = self.external_controller_callback(current_output, current_time)

                # 更新PID参数（用于下一个时间步）
                self.kp = new_kp
                self.ki = new_ki
                self.kd = new_kd

            # --- 5. RK4 积分 ---
            k1_col = self.dt * Xd_col
            k2_col = self.dt * (self.A @ (X_col + k1_col / 2) + self.B * self.u + (F1 + F2) / 2)
            k3_col = self.dt * (self.A @ (X_col + k2_col / 2) + self.B * self.u + (F1 + F2) / 2)
            k4_col = self.dt * (self.A @ (X_col + k3_col) + self.B * self.u + F2)

            X_update_col = X_col + (k1_col + 2 * k2_col + 2 * k3_col + k4_col) / 6.0
            self.X[:, i + 1] = X_update_col.reshape(-1)

            # --- 6. 输出计算 ---
            Y_next = self.C @ X_update_col
            self.Y[i+1] = Y_next.item()

            # 7. 卡尔曼滤波 (Predict & Update)
            # Predict
            X_hat_pred = self.Ad @ X_hat_col + self.Bd * self.u
            P_pred = self.Ad @ self.P @ self.Ad.T + self.Q

            # Update
            S = C_mat @ P_pred @ C_mat.T + R_mat
            K = P_pred @ C_mat.T @ np.linalg.inv(S)

            residual = Y_next - (C_mat @ X_hat_pred)
            X_hat_update = X_hat_pred + K * residual  # 注意这里如果是标量残差，直接乘

            self.P = (np.eye(4) - K @ C_mat) @ P_pred

            self.X_hat[:, i + 1] = X_hat_update.reshape(-1)

            # 记录关键数据：方差
            self.P_diag_history[:, i + 1] = np.diag(self.P)

        # 存储最后一个时间步的PID参数
        self.kp_history[-1] = self.kp
        self.ki_history[-1] = self.ki
        self.kd_history[-1] = self.kd


    def plot_voltage_time(self):
        """
        绘制电压-时间图像。
        时间轴的计算规则：时间 = 索引值 * 0.01
        """
        # 时间轴
        N = self.X.shape[1]
        time_vector = np.arange(N) * self.dt

        # 真实状态组合
        z = -2 * self.X[0, :] + 2 * self.X[1, :]

        # 估计状态组合
        z_hat = -2 * self.X_hat[0, :] + 2 * self.X_hat[1, :]

        # 误差
        z_error = z - z_hat

        plt.figure(figsize=(12, 12))

        # -------- 子图 1：真实状态组合 --------
        plt.subplot(4, 1, 1)
        plt.plot(time_vector, z, label=r'$-2X_0 + 2X_1$')
        plt.ylabel('幅值')
        plt.title('真实状态组合量')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # -------- 子图 2：估计状态组合 --------
        plt.subplot(4, 1, 2)
        plt.plot(time_vector, z_hat, label=r'$-2\hat{X}_0 + 2\hat{X}_1$')
        plt.ylabel('幅值')
        plt.title('估计状态组合量')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # -------- 子图 3：估计误差 --------
        plt.subplot(4, 1, 3)
        plt.plot(time_vector, z_error, label=r'$z - \hat{z}$')
        plt.ylabel('误差')
        plt.title('状态估计误差')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # -------- 子图 4：传感器电压 --------
        plt.subplot(4, 1, 4)
        plt.plot(time_vector, self.Y, label='传感器电压')
        plt.xlabel('时间 (s)')
        plt.ylabel('电压 (V)')
        plt.title('传感器输出')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig('../../results/state_estimation_separate_subplots.jpg')
        plt.show()

    def plot_state_estimation(self):
        N = self.X.shape[1]
        t = np.arange(N) * self.dt

        plt.figure(figsize=(14, 10))

        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.plot(t, self.X[i, :], linewidth=0.5, label=rf'$X_{i}$')
            plt.plot(t, self.X_hat[i, :], '--', linewidth=0.5,
                     label=rf'$\hat{{X}}_{i}$')
            plt.title(rf'状态 $X_{i}$ 与估计')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()

        plt.tight_layout()
        plt.savefig('../../results/state_estimation_all_states.jpg')
        plt.show()

    def plot_state_derivative_estimation(self):
        N = self.Xd.shape[1]
        t = np.arange(N) * self.dt

        plt.figure(figsize=(12, 6))

        for idx, i in enumerate([2, 3]):
            plt.subplot(2, 1, idx + 1)
            plt.plot(t, self.Xd[i, :], linewidth=1.1, label=rf'$\dot X_{i}$')
            plt.plot(t, self.Xd_hat[i, :], '--', linewidth=1.1,
                     label=rf'$\dot{{\hat X}}_{i}$')
            plt.title(rf'导数 $\dot X_{i}$ 与估计')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()

        plt.tight_layout()
        plt.savefig('../../results/state_derivative_estimation.jpg')
        plt.show()


if __name__ == '__main__':
    # set_seed(42)
    config = Config()
    tn = config.EPISODE_LENGTH
    mt = load_mat()
    # 每次循环调用 create_noise_data，RNG 状态不同，噪声也不同
    noise_data = create_noise_data(tn, option='mixed', config=config.SYSTEM_CONFIG, mt_data=mt)
    state_space = StateSpace(config.SYSTEM_CONFIG, noise_data, dt=config.DT, tn=tn, dx=0.005, kp=250, ki=10000, kd=0)
    state_space.solve()
    state_space.plot_voltage_time()
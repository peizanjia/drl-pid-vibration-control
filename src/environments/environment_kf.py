import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # 用于生成源数据表格
import os
from config.config import Config
from ..utils.utils import set_seed, create_noise_data

# 修复 OpenMP 冲突 (Mac/Linux常见)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class StateSpace:
    def __init__(self, config, noise_term, dt=0.01, tn=10000, dx=0.005, kp=0.0, ki=0.0, kd=0.0):
        self.config = config
        self.noise_term = noise_term
        self.dt = dt
        self.tn = tn

        # PID参数
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # PID状态变量
        self.e = 0
        self.e_last = 0
        self.ei = 0
        self.ed = 0

        # --- 系统状态初始化 ---
        self.X = np.zeros((4, tn))
        self.Xd = np.zeros((4, tn))
        self.Y = np.zeros(tn)
        self.u = 0

        # --- 卡尔曼滤波器初始化 ---
        self.X_hat = np.zeros((4, tn))

        # 估计协方差 P
        self.P = np.eye(4) * 0.1
        # 记录 P 的对角线元素 (方差)，用于绘制 3-sigma 包络线
        # 形状: [4, tn], 对应 4 个状态
        self.P_diag_history = np.zeros((4, tn))

        # 过程噪声协方差 Q
        self.Q = np.eye(4) * 1e-5

        # 观测噪声协方差 R
        self.R_val = 1e-3

        self.F1 = noise_term['F1']
        self.F2 = noise_term['F2']

        self.A, self.B, self.C = self.assemble_mat()
        self.Ad = np.eye(4) + self.A * self.dt
        self.Bd = self.B * self.dt

        self.external_controller_callback = None

    def set_external_controller(self, controller_callback):
        self.external_controller_callback = controller_callback

    def assemble_mat(self):
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
            F = np.array([[0], [0], [self.F1[idx, 1]], [self.F2[idx, 1]]])
        else:
            F = np.zeros((4, 1))
        return F

    def solve(self):
        if self.B.ndim == 1: self.B = self.B.reshape(4, 1)
        C_mat = self.C.reshape(1, 4)
        R_mat = np.array([[self.R_val]])

        # 记录初始的 P
        self.P_diag_history[:, 0] = np.diag(self.P)

        for i in range(self.tn - 1):
            # 1. PID 计算 (此处无外部控制时仅为占位)
            self.ei += self.e * self.dt
            self.ed = (self.e - self.e_last) / self.dt
            self.e_last = self.e
            u = self.kp * self.e + self.ki * self.ei + self.kd * self.ed

            X_col = self.X[:, i].reshape(4, 1)
            X_hat_col = self.X_hat[:, i].reshape(4, 1)

            # 2. 物理演化 (RK4)
            F1 = self.__compute_noise(i)
            F2 = self.__compute_noise(i + 1)

            Xd_col = self.A @ X_col - self.B * u + F1
            k1 = self.dt * Xd_col
            k2 = self.dt * (self.A @ (X_col + k1 / 2) + self.B * u + (F1 + F2) / 2)
            k3 = self.dt * (self.A @ (X_col + k2 / 2) + self.B * u + (F1 + F2) / 2)
            k4 = self.dt * (self.A @ (X_col + k3) + self.B * u + F2)

            X_next = X_col + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            self.X[:, i + 1] = X_next.reshape(-1)

            Y_next = (C_mat @ X_next).item()
            if i + 1 < self.tn:
                self.Y[i + 1] = Y_next
                self.e = Y_next

            # 3. 卡尔曼滤波 (Predict & Update)
            # Predict
            X_hat_pred = self.Ad @ X_hat_col + self.Bd * u
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

    def save_source_data(self, output_dir='maturity_level_4_data'):
        """
        保存用于成熟度 4 级证明的 CSV 源数据
        包含：时间、真实状态、估计状态、估计误差、协方差(3-sigma)
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        time_vec = np.arange(self.tn) * self.dt

        # 构建 DataFrame
        data = {'Time': time_vec}
        state_names = ['x1', 'x2', 'v1', 'v2']

        for i in range(4):
            # 真实值
            data[f'{state_names[i]}_True'] = self.X[i, :]
            # 估计值
            data[f'{state_names[i]}_Est'] = self.X_hat[i, :]
            # 误差 (Error = Est - True)
            data[f'{state_names[i]}_Error'] = self.X_hat[i, :] - self.X[i, :]
            # 3-sigma 边界 (用于验证)
            sigma = np.sqrt(self.P_diag_history[i, :])
            data[f'{state_names[i]}_3Sigma'] = 3 * sigma

        df = pd.DataFrame(data)
        file_path = os.path.join(output_dir, 'KF_Validation_SourceData.csv')
        df.to_csv(file_path, index=False)
        print(f"✅ 源数据已保存: {file_path}")
        return df

    def plot_error_analysis(self, save_dir='maturity_level_4_data'):
        """
        绘制专业的 'Estimation Error vs 3-Sigma Bounds' 图
        这是验证 KF 成熟度的标准图表
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        time_vec = np.arange(self.tn) * self.dt
        state_labels = [
            'Modal Displacement 1 ($q_1$)',
            'Modal Displacement 2 ($q_2$)',
            'Modal Velocity 1 ($\dot{q}_1$)',
            'Modal Velocity 2 ($\dot{q}_2$)'
        ]

        fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

        # 定义目标纵轴范围
        Y_LIMIT = 0.1

        for i in range(4):
            ax = axes[i]

            # 计算误差
            error = self.X_hat[i, :] - self.X[i, :]

            # 计算 3-sigma 界限
            sigma = np.sqrt(self.P_diag_history[i, :])
            bound = 3 * sigma

            # 绘制 3-sigma 灰色背景区域 (置信区间)
            ax.fill_between(time_vec, -bound, bound, color='gray', alpha=0.2, label='$3\sigma$ Confidence Interval')
            ax.plot(time_vec, bound, 'k--', linewidth=0.5, alpha=0.5)
            ax.plot(time_vec, -bound, 'k--', linewidth=0.5, alpha=0.5)

            # 绘制实际误差
            ax.plot(time_vec, error, 'b-', linewidth=1, label='Estimation Error ($x_{est} - x_{true}$)')

            # **********************************************
            # 关键修改：设置纵坐标范围
            # **********************************************
            # 仅对 Displacement 和 Velocity 设置范围。
            # 错误 (Error) 的纵坐标通常需要根据实际误差的波动来定，
            # 但如果你需要将误差图也限制在-0.1~0.1，则可以使用以下代码：
            ax.set_ylim(-Y_LIMIT, Y_LIMIT)

            # 统计指标
            rmse = np.sqrt(np.mean(error ** 2))

            ax.set_ylabel(f'Error: {state_labels[i]}', fontsize=10)
            ax.set_title(f'State {i + 1} Estimation Consistency (RMSE = {rmse:.4e})', fontsize=10, pad=3)
            ax.grid(True, linestyle=':', alpha=0.6)

            if i == 0:
                ax.legend(loc='upper right', frameon=True)
            if i == 3:
                ax.set_xlabel('Time (s)', fontsize=12)

        plt.suptitle("Kalman Filter Performance Validation\n(Error must be within $3\sigma$ bounds for consistency)",
                     fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(save_dir, 'KF_Error_Analysis_3Sigma.png')
        plt.savefig(save_path, dpi=300)
        print(f"✅ 分析图表已保存: {save_path}")
        plt.show()


if __name__ == "__main__":
    # 配置与运行
    set_seed(114514)
    config = Config()
    tn = config.EPISODE_LENGTH
    noise_data = create_noise_data(tn)

    # 实例化并求解
    ss = StateSpace(config.SYSTEM_CONFIG, noise_data, dt=config.DT, tn=tn)
    ss.solve()

    # 1. 保存成熟度 4 级所需的源数据 (CSV)
    df = ss.save_source_data()

    # 2. 生成成熟度 4 级所需的分析图 (3-sigma 分析)
    ss.plot_error_analysis()
import numpy as np
from config import config_loader
import matplotlib.pyplot as plt

config = config_loader.load_config()

# X'(t) = AX(t) + Bu(t) + F(t)
# Y(t) = CX(t)

class StateSpace:

    def __init__(self, config, noise_term, dt=0.01, tn=10000, dx=0.005, kp=0, ki=0, kd=0):
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
        self.e_last = 0
        self.ei = 0
        self.ed = 0

        # 状态和输出数组
        self.X = np.zeros((8, tn))
        self.Xd = np.zeros((8, tn))
        self.Y = np.zeros(tn)
        self.u = 0

        # 噪声项
        self.F1 = noise_term['F1']
        self.F2 = noise_term['F2']
        self.F3 = noise_term['F3']
        self.F4 = noise_term['F4']

        # 系统矩阵
        self.A, self.B, self.C = self.assemble_mat()

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
        w3 = self.config['w3']
        w4 = self.config['w4']
        z1 = self.config['z1']
        z2 = self.config['z2']
        z3 = self.config['z3']
        z4 = self.config['z4']
        B1 = self.config['B1']
        B2 = self.config['B2']
        B3 = self.config['B3']
        B4 = self.config['B4']
        C1 = self.config['C1']
        C2 = self.config['C2']
        C3 = self.config['C3']
        C4 = self.config['C4']

        A = np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1],
                      [-w1 ** 2, 0, 0, 0, -2 * z1 * w1, 0, 0, 0],
                      [0, -w2 ** 2, 0, 0, 0, -2 * z2 * w2, 0, 0],
                      [0, 0, -w3 ** 2, 0, 0, 0, -2 * z3 * w3, 0],
                      [0, 0, 0, -w4 ** 2, 0, 0, 0, -2 * z4 * w4], ])

        B = np.array([[0], [0], [0], [0], [B1], [B2], [B3], [B4]])

        C = np.array([C1, C2, C3, C4, 0, 0, 0, 0])

        return A, B, C

    def __compute_noise(self, idx):
        if idx < 1000:
            F = np.array(
                [[0], [0], [0], [0], [self.F1[idx, 1]], [self.F2[idx, 1]], [self.F3[idx, 1]], [self.F4[idx, 1]]]) / 10
        else:
            F = np.zeros((8, 1))
        return F

    def solve(self):
        # 确保矩阵形状正确
        if self.B.ndim == 1 or self.B.shape != (8, 1):
            self.B = self.B.reshape(8, 1)

        # 检查是否设置了外部控制器
        if self.external_controller_callback is None:
            print("警告: 未设置外部控制器，将使用固定PID参数")

        for i in range(self.tn - 1):
            # 存储当前PID参数
            self.kp_history[i] = self.kp
            self.ki_history[i] = self.ki
            self.kd_history[i] = self.kd

            # 0. 关键：将当前状态 X[:, i] 转换为 (8, 1) 列向量
            X_col = self.X[:, i].reshape(8, 1)  # 转换为 (8, 1)

            # --- 1. PID 计算 ---
            self.ei += self.e * self.dt
            self.ed = (self.e - self.e_last) / self.dt
            self.e_last = self.e
            u = self.kp * self.e + self.ki * self.ei + self.kd * self.ed  # u 是标量

            # --- 2. 噪声计算与修正 ---
            F1 = self.__compute_noise(i)
            F2 = self.__compute_noise(i + 1)

            # 确保 F1 和 F2 是 (8, 1)
            if F1.ndim == 1 or F1.shape != (8, 1):
                F1 = F1.reshape(8, 1)
            if F2.ndim == 1 or F2.shape != (8, 1):
                F2 = F2.reshape(8, 1)

            # --- 3. 目标 Xd 赋值 ---
            Xd_col = self.A @ X_col + self.B * u + F1
            self.Xd[:, i] = Xd_col.reshape(-1)

            # --- 4. 输出计算与误差更新 ---
            Y = self.C @ X_col
            self.Y[i] = Y
            self.e = self.Y[i]

            # --- 5. 与外部神经网络控制器交互 ---
            if self.external_controller_callback is not None:
                current_time = i * self.dt
                current_output = self.Y[i]

                # 调用外部控制器获取新的PID参数
                new_kp, new_ki, new_kd = self.external_controller_callback(current_output, current_time)

                # 更新PID参数（用于下一个时间步）
                self.kp = new_kp
                self.ki = new_ki
                self.kd = new_kd

            # --- 6. RK4 积分 ---
            k1_col = self.dt * Xd_col
            k2_col = self.dt * (self.A @ (X_col + k1_col / 2) + self.B * u + (F1 + F2) / 2)
            k3_col = self.dt * (self.A @ (X_col + k2_col / 2) + self.B * u + (F1 + F2) / 2)
            k4_col = self.dt * (self.A @ (X_col + k3_col) + self.B * u + F2)

            X_update_col = X_col + (k1_col + 2 * k2_col + 2 * k3_col + k4_col) / 6.0
            self.X[:, i + 1] = X_update_col.reshape(-1)

        # 存储最后一个时间步的PID参数
        self.kp_history[-1] = self.kp
        self.ki_history[-1] = self.ki
        self.kd_history[-1] = self.kd


    def plot_voltage_time(self):
        """
        绘制电压-时间图像。
        时间轴的计算规则：时间 = 索引值 * 0.01
        """

        # 1. 检查数据是否存在
        if self.Y is None or len(self.Y) == 0:
            print("错误：self.Y 数据为空，无法绘图。")
            return

        N = len(self.Y)

        # 2. 计算时间向量 (X轴数据)
        # 索引值从 0 到 N-1
        indices = np.arange(N)
        # 时间 = 索引值 * 0.01
        time_vector = indices * 0.01

        # 3. 创建电压-时间图像
        plt.figure(figsize=(12, 6))  # 设置图的大小

        # 绘制曲线
        plt.plot(time_vector, self.Y, label="传感器电压")

        # 4. 设置标题和轴标签
        plt.title("传感器电压", fontsize=16)  # 标题
        plt.xlabel("时间 (s)", fontsize=12)  # X轴标签
        plt.ylabel("电压 (V)", fontsize=12)  # Y轴标签

        # 添加图例和网格线
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # 优化 X 轴刻度的显示
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))  # 启用科学计数法显示大数字

        plt.savefig('fixed_control.jpg')

        # 5. 显示图像
        plt.show()
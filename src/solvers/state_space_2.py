import numpy as np
import matplotlib.pyplot as plt
import os


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
        self.Reference = np.zeros(tn)

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

        # 噪声项
        self.F1 = self.noise_term['F1']
        self.F2 = self.noise_term['F2']

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
            self.Xd[:, i] = Xd_col.reshape(-1)

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

        # 存储最后一个时间步的PID参数
        self.kp_history[-1] = self.kp
        self.ki_history[-1] = self.ki
        self.kd_history[-1] = self.kd

    def plot_time_domain_response(self):
        """
        绘制位移与传感器电压的时间响应曲线。
        优化了布局逻辑，增强了学术作图的严谨性。
        """
        # 1. 计算时间轴 (确保 self.dt 已经定义)
        N = self.X.shape[1]
        time_vector = np.arange(N) * self.dt

        # 2. 计算合成位移
        z = -2 * self.X[0, :] + 2 * self.X[1, :]

        # 3. 创建画布：增加 dpi 提升清晰度，预设学术风格
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
        plt.rcParams['axes.unicode_minus'] = False
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # -------- 子图 1：动力学状态 --------
        ax1.plot(time_vector, z, color='#1f77b4', linewidth=1.5, label=r'$z = -2X_0 + 2X_1$')
        ax1.set_ylabel('位移 (m)', fontsize=12)
        ax1.set_title('端部动力学响应 (真实值)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle=':', alpha=0.7)

        # -------- 子图 2：传感器输出 --------
        ax2.plot(time_vector, self.Y.flatten(), color='#d62728', linewidth=1.5, label='Sensor Output')
        ax2.set_xlabel('时间 (s)', fontsize=12)
        ax2.set_ylabel('电压 (V)', fontsize=12)
        ax2.set_title('传感器实测电压信号', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle=':', alpha=0.7)

        # 4. 细节微调
        plt.tight_layout()  # 自动处理子图间距，防止标签重叠

        # 保存时建议使用高质量格式
        plt.savefig('../../results/state_response.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    from config.config import Config
    from config.config_loader import load_mat
    # 1. 基础配置加载
    config = Config()
    tn = config.EPISODE_LENGTH

    # 2. 预加载热学数据
    try:
        mt = load_mat()
    except Exception as e:
        print(f"Warning: mt_data load failed: {e}")
        mt = None

    # 3. 【核心修正】模拟主进程预计算投影系数
    from src.utils.utils import BeamDisturbanceProjector, create_noise_data

    proj = BeamDisturbanceProjector()
    projector_coeffs = proj.get_static_coeffs()

    # 4. 生成噪声数据 (必须传入 projector_data)
    # 你可以尝试修改 option 为 'impact' 来检查冲击是否出现在前半段
    noise_data = create_noise_data(
        tn,
        option='impact',
        system_config=config.SYSTEM_CONFIG,
        mt_data=mt,
        projector_data=projector_coeffs
    )

    # 5. 实例化状态空间求解器
    # 注意：dx 是空间步长，确保它与你的 beam 离散逻辑不冲突
    state_space = StateSpace(
        config.SYSTEM_CONFIG,
        noise_data,
        dt=config.DT,
        tn=tn,
        kp=180,
        ki=10,
        kd=20
    )

    # 6. 求解并绘图
    print("Solving state space equations...")
    state_space.solve()

    print("Plotting results...")
    state_space.plot_time_domain_response()
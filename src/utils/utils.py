import numpy as np
import torch
import random
from scipy.integrate import simpson


def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BeamDisturbanceProjector:
    def __init__(self, L=5.0, D=0.05, thickness=0.002, rho=2800, n_modes=4):
        """
        初始化梁模型参数，预计算模态振型和模态质量
        """
        self.L = L
        self.rho_A = rho * (np.pi * D * thickness)  # 线密度 kg/m
        self.x_mesh = np.linspace(0, L, 500)  # 积分用的离散网格
        self.n_modes = n_modes

        # 悬臂梁特征值 (beta * L)
        self.betas = np.array([1.875, 4.694, 7.855, 10.996])[:n_modes]

        # 预计算模态质量 Mi
        self.Mi = []
        self.phi_funcs = []  # 存储模态函数值用于快速调用

        for i in range(n_modes):
            beta = self.betas[i] / L
            sigma = (np.cosh(beta * L) + np.cos(beta * L)) / (np.sinh(beta * L) + np.sin(beta * L))

            # 模态函数 phi(x)
            phi_x = (np.cosh(beta * self.x_mesh) - np.cos(beta * self.x_mesh)) - \
                    sigma * (np.sinh(beta * self.x_mesh) - np.sin(beta * self.x_mesh))
            self.phi_funcs.append(phi_x)

            # 计算分母：模态质量 integral(rho*A * phi^2)
            mi = simpson(self.rho_A * phi_x ** 2, x=self.x_mesh)
            self.Mi.append(mi)

    def project_jitter(self):
        """
        飞轮微振动 (Base Excitation -> 全域惯性力)
        q(x,t) = -rho*A * a_base(t)
        空间部分 S(x) = -rho*A
        """
        coeffs = []
        for i in range(self.n_modes):
            # 积分: integral(-rho*A * phi_i)
            spatial_integral = simpson(-self.rho_A * self.phi_funcs[i], x=self.x_mesh)
            coeffs.append(spatial_integral / self.Mi[i])
        return np.array(coeffs)

    def project_maneuver(self):
        """
        轨道机动 (全域分布转动惯性力)
        q(x,t) = -rho*A * x * alpha_base(t)
        空间部分 S(x) = -rho*A * x
        """
        coeffs = []
        for i in range(self.n_modes):
            # 积分: integral(-rho*A * x * phi_i)
            spatial_integral = simpson(-self.rho_A * self.x_mesh * self.phi_funcs[i], x=self.x_mesh)
            coeffs.append(spatial_integral / self.Mi[i])
        return np.array(coeffs)

    def project_impact(self, x_loc):
        """
        碎片冲击 (点载荷)
        q(x,t) = F(t) * delta(x - x_loc)
        积分: F(t) * phi_i(x_loc)
        """
        coeffs = []
        # 找到网格中最接近 x_loc 的索引
        idx = (np.abs(self.x_mesh - x_loc)).argmin()

        for i in range(self.n_modes):
            # 采样性质: phi_i(x_loc)
            phi_val = self.phi_funcs[i][idx]
            coeffs.append(phi_val / self.Mi[i])
        return np.array(coeffs)


# 实例化全局投影器 (避免重复计算)
# 假设参数：5m长, 50mm直径铝管
global_projector = BeamDisturbanceProjector(L=5.0, D=0.05, thickness=0.002)


def create_noise_data(tn, dt=0.01, option='normal', config=None, mt_data=None):
    """
    创建具备高度随机性的激励数据 (含 Thermal, Maneuver, Jitter, Impact 复合)
    """
    # --- 类型与模式检查 ---
    if not isinstance(tn, (int, np.integer)):
        raise TypeError(f"参数 'tn' 必须为整数类型, 当前为 {type(tn)}")
    if tn <= 0:
        raise ValueError("参数 'tn' 必须大于 0")

    valid_options = ['normal', 'jitter', 'maneuver', 'impact', 'mixed', 'thermal']
    if option not in valid_options:
        raise ValueError(f"未知的模式选项: '{option}'。支持选项为: {valid_options}")

    time_axis = np.arange(tn) * dt
    modal_forces = np.zeros((tn, 4))

    # 1. 基础底噪 (始终存在)
    base_noise_std = 0.01 if option != 'normal' else 0.1
    for i in range(4):
        modal_forces[:, i] += np.random.normal(0, base_noise_std, tn)

    # 2. 高频微振动 (Jitter)
    if option is not None:
        f_base = np.random.uniform(80, 200)
        amp = np.random.uniform(0.002, 0.008)
        acc_signal = np.zeros(tn)
        harmonics = [1.0, 2.0, 3.5, 4.3]
        for h in harmonics:
            rand_phi = np.random.uniform(0, 2 * np.pi)
            inst_freq = 2 * np.pi * h * f_base * (1 + 0.005 * np.sin(2 * np.pi * 0.2 * time_axis))
            acc_signal += (amp / h) * np.sin(inst_freq * time_axis + rand_phi)

        proj_coeffs = global_projector.project_jitter()
        for i in range(4):
            modal_forces[:, i] += proj_coeffs[i] * acc_signal

    # 3. 轨道机动 (Maneuver)
    if option in ['maneuver', 'mixed']:
        t_start = np.random.uniform(0.1 * tn * dt, 0.4 * tn * dt)
        duration = np.random.uniform(1.5, 3.5)
        max_alpha = np.random.uniform(0.05, 0.15)

        mask = (time_axis >= t_start) & (time_axis < t_start + duration)
        if np.any(mask):
            t_local = time_axis[mask] - t_start
            alpha_t = np.zeros(tn)
            alpha_t[mask] = max_alpha * 0.5 * (1 - np.cos(2 * np.pi * t_local / duration))

            proj_coeffs = global_projector.project_maneuver()
            for i in range(4):
                modal_forces[:, i] += proj_coeffs[i] * alpha_t

    # 4. 碎片冲击 (Impact) - 修正后的叠加逻辑
    if option == 'impact':
        # num_hits = np.random.randint(1, 4)
        num_hits = 1
        for _ in range(num_hits):
            # 随机冲击参数
            t_imp_start = np.random.uniform(0, 0.2 * tn * dt)
            # t_imp_start = 0.1 * tn * dt
            x_loc = np.random.uniform(1.0, 4.8)
            f_peak = np.random.uniform(50, 150)
            dt_imp = 0.1  # 冲击持续时间 100ms

            # 计算冲击对应的索引范围
            start_idx = int(t_imp_start / dt)
            steps_imp = int(dt_imp / dt)
            end_idx = min(start_idx + steps_imp, tn)

            if start_idx < tn:
                # 生成局部脉冲时间轴
                t_local = np.arange(end_idx - start_idx) * dt
                # 半正弦波形
                pulse = f_peak * np.sin(np.pi * t_local / dt_imp)

                # 获取空间投影系数 (请确保 global_projector 内部 Mi 计算正确)
                proj_coeffs = global_projector.project_impact(x_loc)

                for i in range(4):
                    # 【核心修正】：只在冲击发生的特定时间段内进行矢量叠加
                    modal_forces[start_idx:end_idx, i] += proj_coeffs[i] * pulse

    # 5. 热致振动 (Thermal) - 补全逻辑
    if option in ['thermal', 'mixed'] and mt_data is not None:
        if mt_data is not None:
            # 物理逻辑校准：确保 Mt 是一维向量并截取对应长度
            mt_vec = mt_data.flatten()
            if len(mt_vec) < tn:
                raise ValueError(f"[thermal] 数据长度({len(mt_vec)})不足以覆盖仿真时间步数({tn})")

            mt_slice = mt_vec[:tn]

            # 使用 config 中的投影系数 D1~D4 映射到 4 个模态力上
            # 如果 config 为空，则默认不叠加，避免报错
            if config is not None:
                for i in range(4):
                    coeff = config.get(f'D{i + 1}', 0.0)
                    modal_forces[:, i] += coeff * mt_slice
            else:
                # 即使没有 config，如果单独选了 thermal 却没给参数，抛出异常提醒
                if option == 'thermal':
                    raise ValueError("[thermal] 模式需要提供包含 D1~D4 系数的 config 字典")

    # 统一封装返回
    return {f'F{i + 1}': np.column_stack([time_axis, modal_forces[:, i]]) for i in range(4)}
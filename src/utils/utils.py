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
        初始化梁模型参数
        """
        self.L = L
        self.rho_A = rho * (np.pi * D * thickness)
        self.x_mesh = np.linspace(0, L, 500)
        self.n_modes = n_modes
        # 悬臂梁特征值 (beta * L)
        self.betas = np.array([1.875, 4.694, 7.855, 10.996])[:n_modes]

        # 预计算模态函数和质量
        self.Mi = []
        self.phi_funcs = []

        for i in range(n_modes):
            beta = self.betas[i] / L
            sigma = (np.cosh(beta * L) + np.cos(beta * L)) / (np.sinh(beta * L) + np.sin(beta * L))
            phi_x = (np.cosh(beta * self.x_mesh) - np.cos(beta * self.x_mesh)) - \
                    sigma * (np.sinh(beta * self.x_mesh) - np.sin(beta * self.x_mesh))
            self.phi_funcs.append(phi_x)
            mi = simpson(self.rho_A * phi_x ** 2, x=self.x_mesh)
            self.Mi.append(mi)

    def project_jitter(self):
        coeffs = []
        for i in range(self.n_modes):
            spatial_integral = simpson(-self.rho_A * self.phi_funcs[i], x=self.x_mesh)
            coeffs.append(spatial_integral / self.Mi[i])
        return np.array(coeffs)

    def project_maneuver(self):
        coeffs = []
        for i in range(self.n_modes):
            spatial_integral = simpson(-self.rho_A * self.x_mesh * self.phi_funcs[i], x=self.x_mesh)
            coeffs.append(spatial_integral / self.Mi[i])
        return np.array(coeffs)

    def get_phi_at_x(self, x_loc):
        """计算特定位置 x_loc 的模态函数值，用于 Impact"""
        idx = (np.abs(self.x_mesh - x_loc)).argmin()
        return np.array([self.phi_funcs[i][idx] for i in range(self.n_modes)])

    def get_static_coeffs(self):
        """
        【关键修改】
        导出所有静态系数，供并行进程使用，避免传递整个对象。
        """
        return {
            'jitter_coeffs': self.project_jitter(),
            'maneuver_coeffs': self.project_maneuver(),
            'Mi': np.array(self.Mi),
            'phi_funcs_data': self.phi_funcs,  # 如果需要更精细的重建可以传
            'x_mesh': self.x_mesh,
            # 为了 Impact 快速计算，我们保留一个帮助函数或者数据
            # 这里简化处理：因为 Impact 位置随机，我们还是得在 Worker 里算一下插值
            # 或者传入 projector 实例本身（如果 pickle 没问题），或者只传 get_phi_at_x 需要的数据
            # 在 create_noise_data 中，我们将使用近似算法
        }


# --- 功能函数 ---

def create_noise_data(tn, dt=0.01, option='normal', system_config=None, mt_data=None, projector_data=None):
    """
    Args:
        projector_data (dict): 由 BeamDisturbanceProjector.get_static_coeffs() 生成的字典
                               或者包含 project_impact 方法的对象（如果在同一进程）
    """
    # 鲁棒性检查
    if not isinstance(tn, (int, np.integer)) or tn <= 0:
        raise ValueError("tn must be positive integer")

    valid_options = ['normal', 'jitter', 'maneuver', 'impact', 'mixed', 'thermal']
    if option not in valid_options:
        raise ValueError(f"Unknown option: {option}")

    time_axis = np.arange(tn) * dt
    modal_forces = np.zeros((tn, 4))

    # 获取投影系数 (适配并行传递的字典 或 对象)
    # 如果是 dict (并行模式)，则无法调用方法，需要手动处理数据
    # 为了简化，我们假设 projector_data 包含了预计算的向量
    jitter_coeffs = projector_data['jitter_coeffs']
    maneuver_coeffs = projector_data['maneuver_coeffs']
    Mi_vals = projector_data['Mi']

    # ----------------------------------------------------
    # 1. 基础底噪 (Base Noise)
    # ----------------------------------------------------
    base_std = 0.01
    for i in range(4):
        modal_forces[:, i] += np.random.normal(0, base_std, tn)

    # ----------------------------------------------------
    # 2. 强制 Jitter (除去normal模式，始终叠加微振动)
    # ----------------------------------------------------
    # 这里的逻辑修改为：始终存在 Jitter，作为航天器背景
    if option != 'normal':
        f_base = np.random.uniform(80, 200)
        amp = np.random.uniform(0.002, 0.008)
        jitter_signal = np.zeros(tn)
        harmonics = [1.0, 2.0, 3.5, 4.3]
        for h in harmonics:
            rand_phi = np.random.uniform(0, 2 * np.pi)
            inst_freq = 2 * np.pi * h * f_base * (1 + 0.005 * np.sin(2 * np.pi * 0.2 * time_axis))
            jitter_signal += (amp / h) * np.sin(inst_freq * time_axis + rand_phi)

        for i in range(4):
            modal_forces[:, i] += jitter_coeffs[i] * jitter_signal

    # ----------------------------------------------------
    # 3. 模式特定叠加
    # ----------------------------------------------------

    # -> Maneuver
    if option in ['maneuver', 'mixed']:
        t_start = np.random.uniform(0.1, 0.2) * tn * dt
        duration = np.random.uniform(1.5, 3.5)
        max_alpha = np.random.uniform(0.05, 0.15)

        mask = (time_axis >= t_start) & (time_axis < t_start + duration)
        if np.any(mask):
            t_local = time_axis[mask] - t_start
            alpha_t = np.zeros(tn)
            alpha_t[mask] = max_alpha * 0.5 * (1 - np.cos(2 * np.pi * t_local / duration))
            for i in range(4):
                modal_forces[:, i] += maneuver_coeffs[i] * alpha_t

    # -> Impact (物理修正：强制发生在前半段)
    if option == 'impact':
        t_imp_start = np.random.uniform(0.05, 0.10) * tn * dt
        x_loc = np.random.uniform(1.0, 4.8)
        f_peak = np.random.uniform(80, 200)  # 增大一点力度
        dt_imp = 0.1

        start_idx = int(t_imp_start / dt)
        steps_imp = int(dt_imp / dt)
        end_idx = min(start_idx + steps_imp, tn)

        if start_idx < tn:
            t_local = np.arange(end_idx - start_idx) * dt
            pulse = f_peak * np.sin(np.pi * t_local / dt_imp)

            # 计算 impact 投影系数 (内联计算，避免 grid 查找)
            # 近似计算：利用 beam shape function 公式
            # 如果不想传复杂函数，这里需要 projector_data 包含足够信息
            # 简单起见，我们假设 projector_data 里有一个 helper class 或者我们在外面传好了
            # 这里如果不方便插值，可以用简化的形状函数近似

            # 重新实现轻量级 Phi 计算 (不依赖外部对象)
            # 需要 L 和 betas
            L = 5.0  # 硬编码或从 config 传
            betas = np.array([1.875, 4.694, 7.855, 10.996])

            imp_coeffs = []
            for k in range(4):
                beta = betas[k] / L
                bx = beta * x_loc
                bL = betas[k]
                sigma = (np.cosh(bL) + np.cos(bL)) / (np.sinh(bL) + np.sin(bL))
                phi_val = (np.cosh(bx) - np.cos(bx)) - sigma * (np.sinh(bx) - np.sin(bx))
                imp_coeffs.append(phi_val / Mi_vals[k])

            for i in range(4):
                modal_forces[start_idx:end_idx, i] += imp_coeffs[i] * pulse

    # -> Thermal
    if option in ['thermal', 'mixed'] and mt_data is not None:
        if system_config is None:
            # 如果没有 config，且选了 thermal，打印警告或忽略
            pass
        else:
            mt_vec = mt_data.flatten()
            if len(mt_vec) < tn:
                mt_vec = np.tile(mt_vec, int(np.ceil(tn / len(mt_vec))))
            mt_slice = mt_vec[:tn]

            for i in range(4):
                coeff = system_config.get(f'D{i + 1}', 0.0)
                modal_forces[:, i] += coeff * mt_slice

    return {f'F{i + 1}': np.column_stack([time_axis, modal_forces[:, i]]) for i in range(4)}
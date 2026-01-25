import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

# ==========================================
# 1. 核心路径与环境修复 (解决跨目录 Import)
# ==========================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 强制计算 root 目录
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 现在安全导入项目模块
from src.environments.environment_2 import PIDControlEnvironment
from src.agents.ddpg_agent import DDPGAgent
from src.utils.utils import BeamDisturbanceProjector, create_noise_data
from src.solvers.state_space_2 import StateSpace
from config import config


# ==========================================
# 2. 核心评估函数
# ==========================================
def evaluate_performance(agent, config_instance, scenario_settings, model_rel_path, seed=42):
    """
    Args:
        model_rel_path: 相对路径，如 "models/ddpg_ep_70.pth"
    """
    # --- 初始化环境参数 ---
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- 模型加载 (绝对路径保证) ---
    full_model_path = os.path.join(project_root, model_rel_path)
    if os.path.exists(full_model_path):
        agent.load_models(full_model_path)
        agent.actor.eval()
        print(f"[INFO] 成功加载模型权重: {full_model_path}")
    else:
        raise FileNotFoundError(f"找不到模型文件: {full_model_path}")

    # --- 获取外部导入的热数据 (重点) ---
    # 假设 config.py 里通过 config_loader 加载了 THERMAL_MOMENT
    raw_thermal = config_instance.THERMAL_MOMENT
    if isinstance(raw_thermal, dict):
        # 兼容 .mat 导入后的字典格式
        thermal_vector = raw_thermal.get('M_thermal', np.zeros(config_instance.EPISODE_LENGTH)).flatten()
    else:
        thermal_vector = np.array(raw_thermal).flatten()

    # --- 预计算 Beam 静态系数 ---
    proj = BeamDisturbanceProjector(L=5.0, n_modes=4)
    projector_data = proj.get_static_coeffs()

    # 创建保存目录
    save_dir = os.path.join(project_root, "results", "evaluation")
    os.makedirs(save_dir, exist_ok=True)

    # ==========================================
    # 3. 循环测试场景
    # ==========================================
    for scenario_name, settings in scenario_settings.items():
        print(f"\n>>> 正在测试场景: {scenario_name}")

        noise_opt = settings.get('noise_option', 'normal')
        fixed_pid = settings.get('fixed_pid', [80.0, 15.0, 20.0])
        p_range = settings.get('plot_range', [0, 100])
        z_range = settings.get('zoom_range', None)

        # 锁定随机数种子保证公平性
        np.random.seed(seed + 100)

        # ---------------------------------------
        # A. 数据对齐与噪声生成
        # ---------------------------------------
        mt_input = None
        if noise_opt in ['thermal', 'mixed']:
            # 严格截取外部导入的热数据，对齐时间步
            if len(thermal_vector) >= config_instance.EPISODE_LENGTH:
                mt_input = thermal_vector[:config_instance.EPISODE_LENGTH]
            else:
                mt_input = np.pad(thermal_vector, (0, config_instance.EPISODE_LENGTH - len(thermal_vector)))

        noise_data = create_noise_data(
            tn=config_instance.EPISODE_LENGTH,
            dt=config_instance.DT,
            option=noise_opt,
            system_config=config_instance.SYSTEM_CONFIG,
            mt_data=mt_input,
            projector_data=projector_data
        )

        # ---------------------------------------
        # B. RL 自适应仿真
        # ---------------------------------------
        env_rl = PIDControlEnvironment(config_instance)
        # 注入 StateSpace
        ss_rl = StateSpace(
            config_instance.SYSTEM_CONFIG, noise_data,
            dt=config_instance.DT, tn=config_instance.EPISODE_LENGTH,
            kp=config_instance.KP_RANGE[0], ki=config_instance.KI_RANGE[0], kd=config_instance.KD_RANGE[0]
        )
        env_rl.set_state_space(ss_rl)

        obs = env_rl.reset()
        done = False
        rl_log = {'y': [], 'kp': [], 'ki': [], 'kd': []}

        while not done:
            action = agent.select_action(obs, add_noise=False)  # 测试不加噪声
            phys_pid = config_instance.denormalize_action(action)

            rl_log['kp'].append(phys_pid[0])
            rl_log['ki'].append(phys_pid[1])
            rl_log['kd'].append(phys_pid[2])

            obs, _, done, info = env_rl.step(action)
            rl_log['y'].append(info['output'])

        # ---------------------------------------
        # C. 固定 PID 仿真 (Baseline)
        # ---------------------------------------
        env_fix = PIDControlEnvironment(config_instance)
        ss_fix = StateSpace(
            config_instance.SYSTEM_CONFIG, noise_data,
            dt=config_instance.DT, tn=config_instance.EPISODE_LENGTH,
            kp=fixed_pid[0], ki=fixed_pid[1], kd=fixed_pid[2]
        )
        env_fix.set_state_space(ss_fix)

        obs = env_fix.reset()
        done = False
        fix_y = []
        # 固定参数归一化传入 step
        fix_action = config_instance.normalize_action(np.array(fixed_pid))

        while not done:
            _, _, done, info = env_fix.step(fix_action)
            fix_y.append(info['output'])

        # ---------------------------------------
        # D. 专业力学绘图
        # ---------------------------------------
        time = np.arange(len(rl_log['y'])) * config_instance.DT
        mask = (time >= p_range[0]) & (time <= p_range[1])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                       gridspec_kw={'height_ratios': [2, 1]})

        # Subplot 1: 位移响应
        ax1.plot(time[mask], np.array(fix_y)[mask], 'k--', alpha=0.5, label=f'Fixed PID {fixed_pid}')
        ax1.plot(time[mask], np.array(rl_log['y'])[mask], 'r-', linewidth=1.2, label='RL-Adaptive PID')
        ax1.set_ylabel('Sensor Voltage (V)', fontsize=11)
        ax1.set_title(f'Scenario: {scenario_name} (Noise: {noise_opt})', fontweight='bold')
        ax1.legend(loc='upper right', frameon=True)
        ax1.grid(True, which='both', linestyle=':', alpha=0.7)

        # Inset Zoom 局部放大
        if z_range:
            axins = ax1.inset_axes([0.6, 0.15, 0.35, 0.35])
            axins.plot(time, fix_y, 'k--', alpha=0.4)
            axins.plot(time, rl_log['y'], 'r-')
            axins.set_xlim(z_range[0], z_range[1])
            # 自动计算放大区 Y 轴
            z_m = (time >= z_range[0]) & (time <= z_range[1])
            z_vals = np.array(rl_log['y'])[z_m]
            if len(z_vals) > 0:
                axins.set_ylim(np.min(z_vals) * 1.2, np.max(z_vals) * 1.2)
            axins.grid(True)
            ax1.indicate_inset_zoom(axins, edgecolor="black")

        # Subplot 2: Kp 自适应曲线
        ax2.plot(time[mask], np.array(rl_log['kp'])[mask], 'b-', label='Adaptive Kp')
        ax2.set_ylabel('Gain $K_p$', fontsize=11)
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle=':', alpha=0.7)

        plt.tight_layout()
        safe_name = scenario_name.lower().replace(" ", "_")
        plt.savefig(os.path.join(save_dir, f"{safe_name}.png"), dpi=300)
        print(f"[SUCCESS] 图像已保存至: {save_dir}/{safe_name}.png")
        plt.close()


if __name__ == "__main__":
    # 1. 实例化配置
    conf = config.Config()

    # 2. 实例化 Agent
    agent = DDPGAgent(conf)

    # 3. 【核心设置】在这里配置你想测试的所有场景
    # 格式: '图表标题': { 参数... }
    SCENARIO_CONFIG = {
        # 场景 A: 正常工况，画前 20 秒
        "Normal Operation": {
            "noise_option": "normal",
            "fixed_pid": [150.0, 30.0, 20.0],  # 基准 PID
            "plot_range": [0, 10],  # 只画 0-20s
            "zoom_range": None
        },

        "Jitter Operation": {
            "noise_option": "jitter",
            "fixed_pid": [150.0, 30.0, 20.0],  # 基准 PID
            "plot_range": [0, 10],  # 只画 0-20s
            "zoom_range": None
        },

        "Thermal_Stability": {
            "noise_option": "thermal",  # 触发外部 mt_data 逻辑
            "fixed_pid": [150.0, 30.0, 20.0],
            "plot_range": [0, 100],
            "zoom_range": [60, 70]
        },

        # 场景 B: 冲击扰动，画冲击发生的前后
        "Micrometeoroid Impact": {
            "noise_option": "impact",
            "fixed_pid": [150.0, 30.0, 20.0],
            "plot_range": [0, 30],  # 画前50s看收敛
            "zoom_range": [5, 10]  # 假设冲击大概在 5s 左右，放大这里
        },

        # 场景 C: 姿态机动，全过程
        "Slew Maneuver": {
            "noise_option": "maneuver",
            "fixed_pid": [150.0, 30.0, 20.0],  # 也许机动需要软一点的 PID
            "plot_range": [10, 30],  # None 表示画全部 config.EPISODE_LENGTH
            "zoom_range": None  # 放大机动中间段
        },

        # 场景 D: 混合恶劣工况
        "Mixed Extreme": {
            "noise_option": "mixed",
            "fixed_pid": [150.0, 30.0, 20.0],
            "plot_range": [0, 100],
            "zoom_range": [80, 85]
        }
    }

    # 4. 运行评估
    # 确保你有 best_model.pth，或者改为其他路径
    model_path = "models\\ddpg_ep_70.pth"

    evaluate_performance(
        agent=agent,
        config_instance=conf,
        scenario_settings=SCENARIO_CONFIG,
        model_rel_path=model_path,
        seed=123
    )
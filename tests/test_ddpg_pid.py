# root/src/solvers/plot_pid_lqg_drlpid_baseline.py
# Run (from project root):
#   python -m src.solvers.plot_pid_lqg_drlpid_baseline

from __future__ import annotations

import os
import sys
import random
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import zlib

import numpy as np
import torch
import matplotlib.pyplot as plt

# -------------------------
# Path fix
# -------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))  # .../root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# -------------------------
# Project imports
# -------------------------
from config.config import Config
from config.config_loader import load_mat
from src.agents.ddpg_agent import DDPGAgent
from src.utils.utils import BeamDisturbanceProjector, create_noise_data

from src.solvers.state_space_baseline import (
    StateSpaceSimulator, SimConfig,
    make_fixed_pid_controller, make_lqg_controller,
    ControllerBase,
)


# ============================================================
# 1) DRL-PID controller (ENV-consistent 5D obs)
# ============================================================

class DRLPIDController(ControllerBase):
    def __init__(self, agent: DDPGAgent, conf: Config, episode_length: int):
        self.agent = agent
        self.conf = conf
        self.T = episode_length

        self.u_prev = 0.0
        self.e = 0.0
        self.ei = 0.0
        self.ed = 0.0
        self.step_idx = 0

        self.kp_hist = None
        self.ki_hist = None
        self.kd_hist = None

        self._last_pid = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self.u_prev = 0.0
        self.e = 0.0
        self.ei = 0.0
        self.ed = 0.0
        self.step_idx = 0
        self._last_pid = (0.0, 0.0, 0.0)

        self.kp_hist = self.ki_hist = self.kd_hist = None

    def setup_history(self, tn: int) -> None:
        self.kp_hist = np.zeros(tn, dtype=float)
        self.ki_hist = np.zeros(tn, dtype=float)
        self.kd_hist = np.zeros(tn, dtype=float)

    def _obs(self, t_norm: float) -> np.ndarray:
        return np.array([self.u_prev, self.e, self.ei, self.ed, t_norm], dtype=np.float32)

    def log_step(self, i: int) -> None:
        if self.kp_hist is not None:
            kp, ki, kd = self._last_pid
            self.kp_hist[i] = kp
            self.ki_hist[i] = ki
            self.kd_hist[i] = kd

    def export(self) -> Dict[str, Any]:
        return {"kp": self.kp_hist, "ki": self.ki_hist, "kd": self.kd_hist}


class DRLPIDControllerWithC(DRLPIDController):
    def __init__(self, agent: DDPGAgent, conf: Config, episode_length: int, C: np.ndarray):
        super().__init__(agent, conf, episode_length)
        self.C = C.reshape(1, -1)

    def compute(self, *, x: np.ndarray, xdot: np.ndarray, y: float, dt: float, t: float) -> float:
        i = self.step_idx
        if i >= self.T:
            i = self.T - 1
        t_norm = float(i / self.T)

        self.e = float(y)
        self.ei = float(self.ei + self.e * dt)
        self.ed = float((self.C @ np.asarray(xdot).reshape(-1, 1)).item())

        obs = self._obs(t_norm)
        action = self.agent.select_action(obs, add_noise=False)
        pid_phys = self.conf.denormalize_action(action)

        kp, ki, kd = float(pid_phys[0]), float(pid_phys[1]), float(pid_phys[2])
        self._last_pid = (kp, ki, kd)

        u = kp * self.e - ki * self.ei - kd * self.ed

        # if simulator.run() calls set_applied_u, this will be overwritten to saturated u
        self.u_prev = float(u)

        self.step_idx += 1
        return float(u)

    def set_applied_u(self, u_applied: float) -> None:
        self.u_prev = float(u_applied)


# ============================================================
# 2) Plot helpers
# ============================================================

def setup_plot_style():
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.alpha"] = 0.30


def tip_disp(X: np.ndarray) -> np.ndarray:
    return -2.0 * X[0, :] + 2.0 * X[1, :]


def add_inset_multi(
    ax,
    series: List[Tuple[np.ndarray, np.ndarray, str, str, float, int]],
    zoom_range: Tuple[float, float],
    *,
    inset_loc: str = "upper right",
    legend_loc: str = "lower left",
    show_zero_ref: bool = False,
):
    """
    series: list of (t, y, color, label, linewidth, zorder)
    inset 内线条更细，并带 legend
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    axins = inset_axes(ax, width="40%", height="40%", loc=inset_loc, borderpad=1.0)

    for (tt, yy, c, lab, lw, zo) in series:
        axins.plot(tt, yy, color=c, linewidth=lw, label=lab, zorder=zo)

    axins.set_xlim(zoom_range[0], zoom_range[1])

    mask = (series[0][0] >= zoom_range[0]) & (series[0][0] <= zoom_range[1])
    if np.any(mask):
        vals = []
        for (tt, yy, _, _, _, _) in series:
            vals.append(yy[mask])
        yall = np.concatenate(vals)
        pad = 0.10 * (np.max(yall) - np.min(yall) + 1e-12)
        axins.set_ylim(np.min(yall) - pad, np.max(yall) + pad)

    if show_zero_ref:
        axins.axhline(0.0, color="0.3", linestyle="--", linewidth=0.55, alpha=0.8)

    axins.grid(True, linestyle=":", alpha=0.30)
    axins.legend(frameon=False, fontsize=7, loc=legend_loc)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.35", lw=0.8)

def mse_itse_from_window(t: np.ndarray, sig: np.ndarray, plot_range: Tuple[float, float], dt: float) -> Tuple[float, float]:
    """
    Compute MSE and ITSE for signal 'sig' over [plot_range[0], plot_range[1]].
    Error definition: e(t) = sig(t) (ref = 0).
    Time is re-timed: tp = t - plot_range[0].
    ITSE = ∫ tp * e(tp)^2 dt ≈ sum(tp * e^2) * dt
    """
    mask = (t >= plot_range[0]) & (t <= plot_range[1])
    if not np.any(mask):
        return float("nan"), float("nan")

    t0 = float(plot_range[0])
    tp = t[mask] - t0
    e = np.asarray(sig, dtype=float)[mask]

    mse = float(np.mean(e ** 2))
    itse = float(np.sum(tp * (e ** 2)) * dt)
    return mse, itse

def save_three_figs(
    out_dir: Path,
    scenario_key: str,
    t: np.ndarray,
    res_un: Dict[str, Any],
    res_drl: Dict[str, Any],
    res_pid: Dict[str, Any],
    res_lqg: Dict[str, Any],
    plot_range: Tuple[float, float],
    zoom_range: Optional[Tuple[float, float]],
    u_limit: Optional[float] = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- re-time x-axis ---
    mask = (t >= plot_range[0]) & (t <= plot_range[1])
    t0 = float(plot_range[0])
    tp = t[mask] - t0

    Yu, Zu, Uu = res_un["Y"], tip_disp(res_un["X"]), res_un["U"]
    Yd, Zd, Ud = res_drl["Y"], tip_disp(res_drl["X"]), res_drl["U"]
    Yp, Zp, Up = res_pid["Y"], tip_disp(res_pid["X"]), res_pid["U"]
    Yg, Zg, Ug = res_lqg["Y"], tip_disp(res_lqg["X"]), res_lqg["U"]

    # Colors
    c_un  = "0.25"       # gray
    c_drl = "#1f77b4"    # blue
    c_pid = "#ff7f0e"    # orange
    c_lqg = "#d62728"    # red

    # linewidths
    lw_main = 0.9
    lw_un = 0.55  # uncontrolled thinner
    lw_inset = 0.65

    # legend font
    legend_fs = 8

    # Jitter special: control input dense -> half width, and red on top
    jitter_u_dense = (scenario_key.lower() == "jitter")
    if jitter_u_dense:
        lw_u = 0.45  # about half
    else:
        lw_u = lw_main

    # ---------------- Fig1: Uncontrolled + DRL-PID + Large Gain PID ----------------
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True)

    # Control signal (V)
    ax1.plot(tp, Yu[mask], color=c_un, linestyle="--", alpha=0.70, linewidth=lw_un, label="Uncontrolled", zorder=1)
    ax1.plot(tp, Yp[mask], color=c_pid, linestyle="-",  alpha=0.90, linewidth=lw_main, label="Large Gain PID", zorder=2)
    ax1.plot(tp, Yd[mask], color=c_drl, linestyle="-",  alpha=0.98, linewidth=lw_main, label="DRL-PID", zorder=4)  # blue on top
    ax1.axhline(0.0, color="0.3", linestyle="--", linewidth=0.7, alpha=0.8)
    ax1.set_ylabel("Control signal (V)")
    ax1.set_title(f"{scenario_key} | Uncontrolled vs DRL-PID vs Large Gain PID", fontweight="bold")
    ax1.legend(frameon=False, loc="upper left", fontsize=legend_fs)

    # Tip displacement (m)
    ax2.plot(tp, Zu[mask], color=c_un, linestyle="--", alpha=0.70, linewidth=lw_un, label="Uncontrolled", zorder=1)
    ax2.plot(tp, Zp[mask], color=c_pid, linestyle="-",  alpha=0.90, linewidth=lw_main, label="Large Gain PID", zorder=2)
    ax2.plot(tp, Zd[mask], color=c_drl, linestyle="-",  alpha=0.98, linewidth=lw_main, label="DRL-PID", zorder=4)  # blue on top
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Tip displacement (m)")
    ax2.legend(frameon=False, loc="upper left", fontsize=legend_fs)

    if zoom_range is not None:
        zoom_rel = (zoom_range[0] - t0, zoom_range[1] - t0)
        add_inset_multi(
            ax1,
            series=[
                (tp, Yd[mask], c_drl, "DRL-PID", lw_inset, 4),
                (tp, Yp[mask], c_pid, "Large Gain PID", lw_inset, 3),
            ],
            zoom_range=zoom_rel,
            inset_loc="upper right",
            legend_loc="lower left",
            show_zero_ref=True
        )
        add_inset_multi(
            ax2,
            series=[
                (tp, Zd[mask], c_drl, "DRL-PID", lw_inset, 4),
                (tp, Zp[mask], c_pid, "Large Gain PID", lw_inset, 3),
            ],
            zoom_range=zoom_rel,
            inset_loc="upper right",
            legend_loc="lower left",
            show_zero_ref=False
        )

    fig1.tight_layout()
    fig1.savefig(out_dir / f"{scenario_key}_fig1_unctrl_drl_largepid.pdf", bbox_inches="tight")
    plt.close(fig1)

    # ---------------- Fig2: DRL-PID + LQG (NO inset as requested) ----------------
    fig2, (bx1, bx2) = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True)

    bx1.plot(tp, Yg[mask], color=c_lqg, linewidth=lw_main, alpha=0.92, label="LQG", zorder=2)
    bx1.plot(tp, Yd[mask], color=c_drl, linewidth=lw_main, alpha=0.98, label="DRL-PID", zorder=4)  # blue on top
    bx1.axhline(0.0, color="0.3", linestyle="--", linewidth=0.7, alpha=0.8)
    bx1.set_ylabel("Control signal (V)")
    bx1.set_title(f"{scenario_key} | DRL-PID vs LQG", fontweight="bold")
    bx1.legend(frameon=False, loc="upper left", fontsize=legend_fs)

    bx2.plot(tp, Zg[mask], color=c_lqg, linewidth=lw_main, alpha=0.92, label="LQG", zorder=2)
    bx2.plot(tp, Zd[mask], color=c_drl, linewidth=lw_main, alpha=0.98, label="DRL-PID", zorder=4)
    bx2.set_xlabel("Time (s)")
    bx2.set_ylabel("Tip displacement (m)")
    bx2.legend(frameon=False, loc="upper left", fontsize=legend_fs)

    fig2.tight_layout()
    fig2.savefig(out_dir / f"{scenario_key}_fig2_drl_lqg.pdf", bbox_inches="tight")
    plt.close(fig2)

    # ---------------- Fig3: Control input (V) with optional saturation lines ----------------
    fig3, cx = plt.subplots(1, 1, figsize=(6.6, 3.3))

    # draw order: for Jitter, red on top; otherwise blue on top (avoid遮挡)
    if jitter_u_dense:
        cx.plot(tp, Ud[mask], color=c_drl, linewidth=lw_u, alpha=0.85, label="DRL-PID", zorder=2)
        cx.plot(tp, Up[mask], color=c_pid, linewidth=lw_u, alpha=0.85, label="Large Gain PID", zorder=3)
        cx.plot(tp, Ug[mask], color=c_lqg, linewidth=lw_u, alpha=0.95, label="LQG", zorder=5)  # red top
    else:
        cx.plot(tp, Ug[mask], color=c_lqg, linewidth=lw_u, alpha=0.90, label="LQG", zorder=2)
        cx.plot(tp, Up[mask], color=c_pid, linewidth=lw_u, alpha=0.90, label="Large Gain PID", zorder=3)
        cx.plot(tp, Ud[mask], color=c_drl, linewidth=lw_u, alpha=0.98, label="DRL-PID", zorder=5)  # blue top

    cx.set_xlabel("Time (s)")
    cx.set_ylabel("Control input (V)")
    cx.set_title(f"{scenario_key} | Control input", fontweight="bold")
    cx.legend(frameon=False, loc="upper left", fontsize=legend_fs)

    fig3.tight_layout()
    fig3.savefig(out_dir / f"{scenario_key}_fig3_u_compare.pdf", bbox_inches="tight")
    plt.close(fig3)

    # ---------------- Fig4: DRL-PID gains (Kp, Ki, Kd) on one axis ----------------
    ctrl = res_drl.get("controller", {})
    kp_all = ctrl.get("kp", None)
    ki_all = ctrl.get("ki", None)
    kd_all = ctrl.get("kd", None)

    if kp_all is not None and ki_all is not None and kd_all is not None:
        # tp 对应 mask 后的时间轴；kp_all[mask] 对应同一段数据
        tp_full = tp
        kp_seg = kp_all[mask]
        ki_seg = ki_all[mask]
        kd_seg = kd_all[mask]

        # Thermal/Mixed：只画 tp 开始后 10s
        if scenario_key.lower() in ["thermal", "mixed"]:
            pid_mask = (tp_full <= 10.0)
        else:
            pid_mask = np.ones_like(tp_full, dtype=bool)

        fig4, ax = plt.subplots(1, 1, figsize=(6.6, 3.3))

        # 三条线不同颜色（不和前面控制器颜色冲突）
        ax.plot(tp_full[pid_mask], kp_seg[pid_mask], linewidth=0.7, label=r"$K_p$")
        ax.plot(tp_full[pid_mask], ki_seg[pid_mask], linewidth=0.7, label=r"$K_i$")
        ax.plot(tp_full[pid_mask], kd_seg[pid_mask], linewidth=0.7, label=r"$K_d$")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("PID gains")
        ax.set_title(f"{scenario_key} | DRL-PID gains", fontweight="bold")
        ax.legend(frameon=False, fontsize=8, loc="upper right")

        fig4.tight_layout()
        fig4.savefig(out_dir / f"{scenario_key}_fig4_drl_pid_gains.pdf", bbox_inches="tight")
        plt.close(fig4)


# ============================================================
# 3) Main
# ============================================================

def main():
    setup_plot_style()

    SEED = 123
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    conf = Config()
    tn = conf.EPISODE_LENGTH
    dt = conf.DT

    # Agent
    agent = DDPGAgent(conf)

    model_rel = r"models\ddpg_ep_63.pth"
    model_path = os.path.join(project_root, model_rel)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    agent.load_models(model_path)
    agent.actor.eval()
    print(f"[INFO] Loaded model: {model_path}")

    # thermal data
    try:
        _ = load_mat()
    except Exception:
        pass

    # projector
    proj = BeamDisturbanceProjector(L=5.0, n_modes=4)
    projector_data = proj.get_static_coeffs()

    out_dir = Path(project_root) / "results" / "baseline_pid_lqg_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    # keep your modified plot ranges / zoom ranges
    scenarios = {
        "Jitter":   {"noise_option": "jitter",   "fixed_pid": [150.0, 30.0, 20.0], "plot_range": (0.0, 10.0),  "zoom_range": None},
        "Thermal":  {"noise_option": "thermal",  "fixed_pid": [150.0, 30.0, 20.0], "plot_range": (0.0, 70.0), "zoom_range": (65.0, 70.0)},
        "Impact":   {"noise_option": "impact",   "fixed_pid": [150.0, 30.0, 20.0], "plot_range": (5.0, 15.0),  "zoom_range": None},
        "Maneuver": {"noise_option": "maneuver", "fixed_pid": [150.0, 30.0, 20.0], "plot_range": (18.0, 23.0), "zoom_range": None},
        "Mixed":    {"noise_option": "mixed",    "fixed_pid": [150.0, 30.0, 20.0], "plot_range": (0.0, 70.0), "zoom_range": (65.0, 70.0)},
    }

    # LQG hyperparameters
    Q_lqr = np.diag([1, 1, 10, 10])
    R_lqr = np.array([[1e-7]])
    W_kf = np.diag([1e-6, 1e-6, 1e-3, 1e-3])
    V_kf = np.array([[1e-4]])

    # thermal vector extraction
    thermal_vector = None
    raw_thermal = getattr(conf, "THERMAL_MOMENT", None)
    if raw_thermal is not None:
        if isinstance(raw_thermal, dict):
            thermal_vector = raw_thermal.get("M_thermal", np.zeros(tn)).flatten()
        else:
            thermal_vector = np.array(raw_thermal).flatten()

    U_LIMIT = 300.0

    for scen_name, cfg in scenarios.items():
        noise_opt = cfg["noise_option"]
        fixed_pid = cfg["fixed_pid"]
        plot_range = cfg["plot_range"]
        zoom_range = cfg["zoom_range"]

        scen_seed = (SEED + zlib.crc32(scen_name.encode("utf-8")) % 100000) % (2**32 - 1)
        np.random.seed(scen_seed)
        random.seed(scen_seed)
        torch.manual_seed(scen_seed)

        mt_input = None
        if noise_opt in ["thermal", "mixed"] and thermal_vector is not None:
            if len(thermal_vector) >= tn:
                mt_input = thermal_vector[:tn]
            else:
                mt_input = np.pad(thermal_vector, (0, tn - len(thermal_vector)))

        noise_data = create_noise_data(
            tn=tn,
            dt=dt,
            option=noise_opt,
            system_config=conf.SYSTEM_CONFIG,
            mt_data=mt_input,
            projector_data=projector_data
        )

        sim = StateSpaceSimulator(
            system_config=conf.SYSTEM_CONFIG,
            noise_term=noise_data,
            sim_cfg=SimConfig(dt=dt, tn=tn, u_limit=U_LIMIT, meas_noise_std=0.0)
        )

        ctrl_un = make_fixed_pid_controller(conf.SYSTEM_CONFIG, kp=0.0, ki=0.0, kd=0.0)
        ctrl_drl = DRLPIDControllerWithC(agent=agent, conf=conf, episode_length=tn, C=sim.C)
        ctrl_pid = make_fixed_pid_controller(conf.SYSTEM_CONFIG, kp=fixed_pid[0], ki=fixed_pid[1], kd=fixed_pid[2])
        ctrl_lqg = make_lqg_controller(conf.SYSTEM_CONFIG, dt=dt, Q_lqr=Q_lqr, R_lqr=R_lqr, W_kf=W_kf, V_kf=V_kf)

        res_un = sim.run(ctrl_un)
        res_drl = sim.run(ctrl_drl)
        res_pid = sim.run(ctrl_pid)
        res_lqg = sim.run(ctrl_lqg)

        # ---------------- Metrics: MSE / ITSE (based on control signal Y, ref=0) ----------------
        t = res_un["t"]

        mse_un,  itse_un  = mse_itse_from_window(t, res_un["Y"],  plot_range, dt)
        mse_drl, itse_drl = mse_itse_from_window(t, res_drl["Y"], plot_range, dt)
        mse_pid, itse_pid = mse_itse_from_window(t, res_pid["Y"], plot_range, dt)
        mse_lqg, itse_lqg = mse_itse_from_window(t, res_lqg["Y"], plot_range, dt)

        print(f"\n[{scen_name}] Metrics on Control signal Y (window {plot_range[0]}–{plot_range[1]} s, retimed tp):")
        print(f"  Uncontrolled   : MSE={mse_un:.6e} | ITSE={itse_un:.6e}")
        print(f"  DRL-PID        : MSE={mse_drl:.6e} | ITSE={itse_drl:.6e}")
        print(f"  Large Gain PID : MSE={mse_pid:.6e} | ITSE={itse_pid:.6e}")
        print(f"  LQG            : MSE={mse_lqg:.6e} | ITSE={itse_lqg:.6e}")

        t = res_un["t"]
        save_three_figs(
            out_dir=out_dir,
            scenario_key=scen_name,
            t=t,
            res_un=res_un,
            res_drl=res_drl,
            res_pid=res_pid,
            res_lqg=res_lqg,
            plot_range=plot_range,
            zoom_range=zoom_range,
            u_limit=U_LIMIT
        )
        print(f"[OK] {scen_name} -> {out_dir}")

    print(f"\nAll done. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
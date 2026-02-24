# root/src/solvers/plot_pid_lqg_drlpid_baseline.py
# Run (from project root):
#   python -m src.solvers.plot_pid_lqg_drlpid_baseline
#
# Prerequisites:
# 1) You have UPDATED root/src/solvers/state_space_baseline.py:
#    - StateSpaceSimulator.run() uses env-consistent timing:
#      Xd_col with u_old, ed based on that, u_new computed, RK4 k1 uses u_old, k2-4 use u_new.
#    - (optional) run() calls controller.log_step(i) if available (the patch I provided).
#
# 2) Your trained model exists, e.g. models/ddpg_ep_70.pth
#
# Outputs:
# results/baseline_pid_lqg_compare/<scenario>_fig{1,2,3}_*.pdf (vector)

from __future__ import annotations

import os
import sys
import random
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

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
# 1) DRL-PID controller (ENV-consistent 5D obs):
#    state = [u_prev, e, ei, ed, t_norm]
# ============================================================

class DRLPIDController(ControllerBase):
    """
    Uses a trained DDPG actor to output PID params. Obs matches PIDControlEnvironment:

      s_i = [u_{i-1}, e_i, ei_i, ed_i, i/T]

    Timing alignment:
      - y_now, xdot (based on u_old) are provided by simulator.run() BEFORE controller.compute()
      - controller.compute() updates (e, ei, ed) using these and produces u_new
      - u_prev for next step is set to u_new (stored internally)
    """

    def __init__(self, agent: DDPGAgent, conf: Config, episode_length: int):
        self.agent = agent
        self.conf = conf
        self.T = episode_length

        # internal cached signals
        self.u_prev = 0.0
        self.e = 0.0
        self.ei = 0.0
        self.ed = 0.0
        self.step_idx = 0

        # histories
        self.kp_hist = None
        self.ki_hist = None
        self.kd_hist = None
        self.u_hist = None
        self.e_hist = None
        self.ei_hist = None
        self.ed_hist = None

        self._last_pid = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self.u_prev = 0.0
        self.e = 0.0
        self.ei = 0.0
        self.ed = 0.0
        self.step_idx = 0
        self._last_pid = (0.0, 0.0, 0.0)

        self.kp_hist = self.ki_hist = self.kd_hist = None
        self.u_hist = None
        self.e_hist = self.ei_hist = self.ed_hist = None

    def setup_history(self, tn: int) -> None:
        self.kp_hist = np.zeros(tn, dtype=float)
        self.ki_hist = np.zeros(tn, dtype=float)
        self.kd_hist = np.zeros(tn, dtype=float)
        self.u_hist = np.zeros(tn, dtype=float)
        self.e_hist = np.zeros(tn, dtype=float)
        self.ei_hist = np.zeros(tn, dtype=float)
        self.ed_hist = np.zeros(tn, dtype=float)

    def _obs(self, t_norm: float) -> np.ndarray:
        return np.array([self.u_prev, self.e, self.ei, self.ed, t_norm], dtype=np.float32)

    def compute(self, *, x: np.ndarray, xdot: np.ndarray, y: float, dt: float, t: float) -> float:
        # env-consistent: at step i, simulator has computed xdot using u_old and passed it in.
        # In env: e=y_now, ei += e*dt, ed = C@Xd_col. Here ed is already computed upstream as C@Xd_col
        # BUT simulator passes full xdot, not ed. So we must compute ed = C@xdot? No: env stores ed = (C@Xd_col).item().
        # We can emulate env by recomputing ed as dot(C, xdot) IF C is available.
        # In this controller, we assume ed is already embedded in y? Not.
        # Solution: use env definition: ed := derivative of y = C xdot. We can compute it only if we know C.
        # Since env uses that, we fetch C from config.SYSTEM_CONFIG at init? Not available here.
        # Safer: approximate ed with ydot = y - y_prev over dt? Not env-consistent.
        #
        # Therefore: this DRL controller should be constructed with C, or simulator should pass ed directly.
        # We take the pragmatic route: compute ed as (y - y_prev)/dt using internal y_prev.
        # But to remain STRICTLY env-consistent, pass C into this controller and compute ed = C@xdot.
        raise RuntimeError(
            "DRLPIDController needs measurement derivative ed consistent with env: ed = C@xdot(u_old). "
            "Please construct DRLPIDController with system C and compute ed = C @ xdot."
        )

    def log_step(self, i: int) -> None:
        if self.u_hist is not None:
            kp, ki, kd = self._last_pid
            self.kp_hist[i] = kp
            self.ki_hist[i] = ki
            self.kd_hist[i] = kd
            self.u_hist[i] = self.u_prev
            self.e_hist[i] = self.e
            self.ei_hist[i] = self.ei
            self.ed_hist[i] = self.ed

    def export(self) -> Dict[str, Any]:
        return {
            "kp": self.kp_hist, "ki": self.ki_hist, "kd": self.kd_hist,
            "u_prev": self.u_hist,
            "e": self.e_hist, "ei": self.ei_hist, "ed": self.ed_hist,
        }


# ---------- Correct env-consistent DRL controller with C ----------
class DRLPIDControllerWithC(DRLPIDController):
    def __init__(self, agent: DDPGAgent, conf: Config, episode_length: int, C: np.ndarray):
        super().__init__(agent, conf, episode_length)
        self.C = C.reshape(1, -1)

    def compute(self, *, x: np.ndarray, xdot: np.ndarray, y: float, dt: float, t: float) -> float:
        i = self.step_idx
        if i >= self.T:
            i = self.T - 1
        t_norm = float(i / self.T)  # env: i / episode_length

        # env-consistent internal update
        self.e = float(y)
        self.ei = float(self.ei + self.e * dt)
        self.ed = (self.C @ xdot.reshape(-1, 1)).item()  # ed = C * Xd_col(u_old)

        obs = self._obs(t_norm)
        action = self.agent.select_action(obs, add_noise=False)
        pid_phys = self.conf.denormalize_action(action)

        kp, ki, kd = float(pid_phys[0]), float(pid_phys[1]), float(pid_phys[2])
        self._last_pid = (kp, ki, kd)

        u = kp * self.e - ki * self.ei - kd * self.ed

        # shift for next step (u_prev = u_new)
        self.u_prev = float(u)
        self.step_idx += 1
        return float(u)


# ============================================================
# 2) Plot helpers (vector PDF)
# ============================================================

def setup_plot_style():
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["lines.linewidth"] = 1.0
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.alpha"] = 0.35


def tip_disp(X: np.ndarray) -> np.ndarray:
    return -2.0 * X[0, :] + 2.0 * X[1, :]


def add_inset(ax, t, y, zoom_range: Tuple[float, float], loc="upper right"):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    axins = inset_axes(ax, width="38%", height="38%", loc=loc, borderpad=1.0)
    axins.plot(t, y)
    axins.set_xlim(zoom_range[0], zoom_range[1])

    mask = (t >= zoom_range[0]) & (t <= zoom_range[1])
    if np.any(mask):
        yy = y[mask]
        pad = 0.08 * (np.max(yy) - np.min(yy) + 1e-12)
        axins.set_ylim(np.min(yy) - pad, np.max(yy) + pad)

    axins.grid(True, linestyle=":", alpha=0.35)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.3", lw=0.8)


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
):
    out_dir.mkdir(parents=True, exist_ok=True)

    mask = (t >= plot_range[0]) & (t <= plot_range[1])

    Yu, Zu, Uu = res_un["Y"], tip_disp(res_un["X"]), res_un["U"]
    Yd, Zd, Ud = res_drl["Y"], tip_disp(res_drl["X"]), res_drl["U"]
    Yp, Zp, Up = res_pid["Y"], tip_disp(res_pid["X"]), res_pid["U"]
    Yg, Zg, Ug = res_lqg["Y"], tip_disp(res_lqg["X"]), res_lqg["U"]

    c_un  = "0.25"
    c_drl = "#1f77b4"
    c_pid = "#ff7f0e"
    c_lqg = "#2ca02c"

    # Fig1
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True)
    ax1.plot(t[mask], Yu[mask], color=c_un, linestyle="--", alpha=0.75, label="Uncontrolled (PID=0)")
    ax1.plot(t[mask], Yd[mask], color=c_drl, linestyle="-", alpha=0.95, label="DRL-PID")
    ax1.set_ylabel("Sensor voltage $y$")
    ax1.set_title(f"{scenario_key} | Uncontrolled vs DRL-PID", fontweight="bold")
    ax1.legend(frameon=False, loc="upper right")

    ax2.plot(t[mask], Zu[mask], color=c_un, linestyle="--", alpha=0.75, label="Uncontrolled (PID=0)")
    ax2.plot(t[mask], Zd[mask], color=c_drl, linestyle="-", alpha=0.95, label="DRL-PID")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel(r"Tip disp. $z=-2\eta_1+2\eta_2$")
    ax2.legend(frameon=False, loc="upper right")

    if zoom_range is not None:
        add_inset(ax1, t, Yd, zoom_range)
        add_inset(ax2, t, Zd, zoom_range)

    fig1.tight_layout()
    fig1.savefig(out_dir / f"{scenario_key}_fig1_unctrl_vs_drl.pdf", bbox_inches="tight")
    plt.close(fig1)

    # Fig2
    fig2, (bx1, bx2) = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True)
    bx1.plot(t[mask], Yd[mask], color=c_drl, label="DRL-PID")
    bx1.plot(t[mask], Yp[mask], color=c_pid, label="Large Gain PID")
    bx1.plot(t[mask], Yg[mask], color=c_lqg, label="LQG")
    bx1.set_ylabel("Sensor voltage $y$")
    bx1.set_title(f"{scenario_key} | DRL-PID vs Large Gain PID vs LQG", fontweight="bold")
    bx1.legend(frameon=False, loc="upper right")

    bx2.plot(t[mask], Zd[mask], color=c_drl, label="DRL-PID")
    bx2.plot(t[mask], Zp[mask], color=c_pid, label="Large Gain PID")
    bx2.plot(t[mask], Zg[mask], color=c_lqg, label="LQG")
    bx2.set_xlabel("Time (s)")
    bx2.set_ylabel(r"Tip disp. $z=-2\eta_1+2\eta_2$")
    bx2.legend(frameon=False, loc="upper right")

    if zoom_range is not None:
        add_inset(bx1, t, Yd, zoom_range)
        add_inset(bx2, t, Zd, zoom_range)

    fig2.tight_layout()
    fig2.savefig(out_dir / f"{scenario_key}_fig2_drl_pid_lqg.pdf", bbox_inches="tight")
    plt.close(fig2)

    # Fig3
    fig3, cx = plt.subplots(1, 1, figsize=(6.6, 3.3))
    cx.plot(t[mask], Ud[mask], color=c_drl, label="DRL-PID")
    cx.plot(t[mask], Up[mask], color=c_pid, label="Large Gain PID")
    cx.plot(t[mask], Ug[mask], color=c_lqg, label="LQG")
    cx.set_xlabel("Time (s)")
    cx.set_ylabel("Control voltage $u$")
    cx.set_title(f"{scenario_key} | Control input", fontweight="bold")
    cx.legend(frameon=False, loc="upper right")

    if zoom_range is not None:
        add_inset(cx, t, Ud, zoom_range)

    fig3.tight_layout()
    fig3.savefig(out_dir / f"{scenario_key}_fig3_u_compare.pdf", bbox_inches="tight")
    plt.close(fig3)


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

    model_rel = r"models\ddpg_ep_100.pth"
    model_path = os.path.join(project_root, model_rel)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    agent.load_models(model_path)
    agent.actor.eval()
    print(f"[INFO] Loaded model: {model_path}")

    # thermal data
    try:
        mt_loaded = load_mat()
    except Exception:
        mt_loaded = None

    # projector
    proj = BeamDisturbanceProjector(L=5.0, n_modes=4)
    projector_data = proj.get_static_coeffs()

    out_dir = Path(project_root) / "results" / "baseline_pid_lqg_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    # exactly 5 scenarios
    scenarios = {
        "Jitter":   {"noise_option": "jitter",   "fixed_pid": [150.0, 30.0, 20.0], "plot_range": (0.0, 10.0),  "zoom_range": None},
        "Thermal":  {"noise_option": "thermal",  "fixed_pid": [150.0, 30.0, 20.0], "plot_range": (0.0, 100.0), "zoom_range": (60.0, 70.0)},
        "Impact":   {"noise_option": "impact",   "fixed_pid": [150.0, 30.0, 20.0], "plot_range": (5.0, 30.0),  "zoom_range": (5.0, 15.0)},
        "Maneuver": {"noise_option": "maneuver", "fixed_pid": [150.0, 30.0, 20.0], "plot_range": (10.0, 30.0), "zoom_range": None},
        "Mixed":    {"noise_option": "mixed",    "fixed_pid": [150.0, 30.0, 20.0], "plot_range": (0.0, 100.0), "zoom_range": (60.0, 70.0)},
    }

    # LQG hyperparameters
    Q_lqr = np.diag([1, 1, 10, 10])
    R_lqr = np.array([[1e-7]])
    W_kf = np.diag([1e-6, 1e-6, 1e-3, 1e-3])
    V_kf = np.array([[1e-4]])

    # thermal vector extraction (match your original evaluation)
    thermal_vector = None
    raw_thermal = getattr(conf, "THERMAL_MOMENT", None)
    if raw_thermal is not None:
        if isinstance(raw_thermal, dict):
            thermal_vector = raw_thermal.get("M_thermal", np.zeros(tn)).flatten()
        else:
            thermal_vector = np.array(raw_thermal).flatten()

    for scen_name, cfg in scenarios.items():
        noise_opt = cfg["noise_option"]
        fixed_pid = cfg["fixed_pid"]
        plot_range = cfg["plot_range"]
        zoom_range = cfg["zoom_range"]

        # per-scenario deterministic seed
        scen_seed = (SEED + (abs(hash(scen_name)) % 100000)) % (2**32 - 1)
        np.random.seed(scen_seed)
        random.seed(scen_seed)
        torch.manual_seed(scen_seed)

        # mt_input only for thermal/mixed
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
            sim_cfg=SimConfig(dt=dt, tn=tn, u_limit=None, meas_noise_std=0.0)
        )

        # controllers
        ctrl_un = make_fixed_pid_controller(conf.SYSTEM_CONFIG, kp=0.0, ki=0.0, kd=0.0)

        ctrl_drl = DRLPIDControllerWithC(agent=agent, conf=conf, episode_length=tn, C=sim.C)

        ctrl_pid = make_fixed_pid_controller(conf.SYSTEM_CONFIG, kp=fixed_pid[0], ki=fixed_pid[1], kd=fixed_pid[2])

        ctrl_lqg = make_lqg_controller(
            conf.SYSTEM_CONFIG, dt=dt,
            Q_lqr=Q_lqr, R_lqr=R_lqr,
            W_kf=W_kf, V_kf=V_kf
        )

        # run (same noise_data)
        res_un = sim.run(ctrl_un)
        res_drl = sim.run(ctrl_drl)
        res_pid = sim.run(ctrl_pid)
        res_lqg = sim.run(ctrl_lqg)

        # save figs (vector PDF)
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
            zoom_range=zoom_range
        )
        print(f"[OK] {scen_name} -> {out_dir}")

    print(f"\nAll done. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
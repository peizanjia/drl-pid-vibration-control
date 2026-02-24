# simulator.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple


# ============================================================
#  LQR / Kalman helpers
# ============================================================

def _solve_care_fallback(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
                        max_iter: int = 300, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Continuous-time CARE fallback (Kleinman-like). Prefer SciPy if available.
    Returns (P, K) for u = -Kx.
    """
    n = A.shape[0]
    Rinv = np.linalg.inv(R)

    K = np.zeros((B.shape[1], n))
    P = Q.copy()

    for _ in range(max_iter):
        Acl = A - B @ K
        S = Q + K.T @ R @ K

        # Solve Lyapunov: Acl^T P + P Acl + S = 0  via vec
        M = np.kron(np.eye(n), Acl.T) + np.kron(Acl.T, np.eye(n))
        vecP = np.linalg.solve(M, (-S).reshape(-1, order="F"))
        Pn = vecP.reshape(n, n, order="F")
        Pn = 0.5 * (Pn + Pn.T)

        Kn = Rinv @ (B.T @ Pn)

        if np.linalg.norm(Pn - P, ord="fro") < tol:
            return Pn, Kn
        P, K = Pn, Kn

    return P, K


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Continuous-time LQR gain. Returns (P, K) for u=-Kx."""
    try:
        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.solve(R, B.T @ P)
        return P, K
    except Exception:
        return _solve_care_fallback(A, B, Q, R)


def discretize_zoh(A: np.ndarray, B: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    ZOH discretization:
      x_{k+1} = Ad x_k + Bd u_k
    Prefer SciPy expm; fallback uses Euler.
    """
    try:
        from scipy.linalg import expm
        n = A.shape[0]
        m = B.shape[1]
        M = np.zeros((n + m, n + m), dtype=float)
        M[:n, :n] = A
        M[:n, n:] = B
        Md = expm(M * dt)
        Ad = Md[:n, :n]
        Bd = Md[:n, n:]
        return Ad, Bd
    except Exception:
        n = A.shape[0]
        Ad = np.eye(n) + A * dt
        Bd = B * dt
        return Ad, Bd


def dlqe_steady_gain(Ad: np.ndarray, Cd: np.ndarray, W: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Steady-state discrete Kalman gain:
      P = Ad P Ad^T - Ad P Cd^T (Cd P Cd^T + V)^{-1} Cd P Ad^T + W
      L = P Cd^T (Cd P Cd^T + V)^{-1}
    Returns (P, L).
    """
    try:
        from scipy.linalg import solve_discrete_are
        P = solve_discrete_are(Ad.T, Cd.T, W, V)
        S = Cd @ P @ Cd.T + V
        L = P @ Cd.T @ np.linalg.inv(S)
        return P, L
    except Exception:
        P = W.copy()
        for _ in range(2000):
            S = Cd @ P @ Cd.T + V
            Kg = P @ Cd.T @ np.linalg.inv(S)
            Pn = Ad @ P @ Ad.T - Ad @ Kg @ Cd @ P @ Ad.T + W
            Pn = 0.5 * (Pn + Pn.T)
            if np.linalg.norm(Pn - P, ord="fro") < 1e-10:
                P = Pn
                break
            P = Pn
        S = Cd @ P @ Cd.T + V
        L = P @ Cd.T @ np.linalg.inv(S)
        return P, L


# ============================================================
#  System assembly + noise injection (match your convention)
# ============================================================

def assemble_matrices(config: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w1 = config['w1']
    w2 = config['w2']
    z1 = config['z1']
    z2 = config['z2']
    B1 = config['B1']
    B2 = config['B2']
    C1 = config['C1']
    C2 = config['C2']

    A = np.array([[0,        0,        1,            0           ],
                  [0,        0,        0,            1           ],
                  [-w1 ** 2, 0,        -2 * z1 * w1, 0           ],
                  [0,        -w2 ** 2, 0,            -2 * z2 * w2]], dtype=float)

    B = np.array([[0], [0], [B1], [B2]], dtype=float)
    C = np.array([C1, C2, 0, 0], dtype=float)
    return A, B, C


def compute_disturbance(noise_term: Dict[str, np.ndarray], idx: int, tn: int) -> np.ndarray:
    """
    Returns F as (4,1): [[0],[0],[F1[idx,1]],[F2[idx,1]]]
    """
    if idx < tn:
        F1 = noise_term['F1']
        F2 = noise_term['F2']
        return np.array([[0.0], [0.0], [float(F1[idx, 1])], [float(F2[idx, 1])]], dtype=float)
    return np.zeros((4, 1), dtype=float)


# ============================================================
#  Controllers
# ============================================================

class ControllerBase:
    def reset(self) -> None:
        pass

    def compute(self, *, x: np.ndarray, xdot: np.ndarray, y: float, dt: float, t: float) -> float:
        raise NotImplementedError

    def export(self) -> Dict[str, Any]:
        """Optional extra signals for logging."""
        return {}


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float


class FixedPIDController(ControllerBase):
    """
    Fixed PID using measurement y and derivative from xdot:
      e = y
      ei += e*dt
      ed = C * xdot
      u = kp*e - ki*ei - kd*ed
    """
    def __init__(self, gains: PIDGains, C: np.ndarray):
        self.gains = gains
        self.C = C.reshape(1, -1)
        self.e = 0.0
        self.ei = 0.0
        self.ed = 0.0

        self.e_hist = None
        self.ei_hist = None
        self.ed_hist = None

    def reset(self) -> None:
        self.e = 0.0
        self.ei = 0.0
        self.ed = 0.0
        self.e_hist = None
        self.ei_hist = None
        self.ed_hist = None

    def setup_history(self, tn: int) -> None:
        self.e_hist = np.zeros(tn, dtype=float)
        self.ei_hist = np.zeros(tn, dtype=float)
        self.ed_hist = np.zeros(tn, dtype=float)

    def compute(self, *, x: np.ndarray, xdot: np.ndarray, y: float, dt: float, t: float) -> float:
        self.e = float(y)
        self.ei += self.e * dt
        self.ed = (self.C @ xdot.reshape(-1, 1)).item()

        u = (self.gains.kp * self.e) - (self.gains.ki * self.ei) - (self.gains.kd * self.ed)
        return float(u)

    def log_step(self, i: int) -> None:
        if self.e_hist is not None:
            self.e_hist[i] = self.e
            self.ei_hist[i] = self.ei
            self.ed_hist[i] = self.ed

    def export(self) -> Dict[str, Any]:
        return {"e": self.e_hist, "ei": self.ei_hist, "ed": self.ed_hist}


class LQGController(ControllerBase):
    """
    LQG controller:
      - LQR: u = -K xhat
      - Kalman (discrete): xhat_{k+1} = Ad xhat_k + Bd u_k + Kg (y_{k+1} - C xhat_pred)

    Notes:
      - We use measurement y each step; by default we update using y_next (after plant integration).
      - Process noise covariance W_kf and measurement covariance V_kf are "hyperparameters".
    """
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, dt: float,
                 Q_lqr: np.ndarray, R_lqr: np.ndarray,
                 W_kf: np.ndarray, V_kf: np.ndarray):
        self.A = A
        self.B = B
        self.C = C.reshape(1, -1)
        self.dt = dt

        self.Q_lqr = Q_lqr
        self.R_lqr = R_lqr
        self.W_kf = W_kf
        self.V_kf = V_kf

        # LQR gain
        self.P_lqr, self.K = lqr_gain(self.A, self.B, self.Q_lqr, self.R_lqr)

        # Discretize for Kalman
        self.Ad, self.Bd = discretize_zoh(self.A, self.B, self.dt)
        self.Cd = self.C.copy()

        # Steady-state Kalman gain (for reference / init)
        self.P_ss, self.L_ss = dlqe_steady_gain(self.Ad, self.Cd, self.W_kf, self.V_kf)

        # Runtime filter state/cov
        self.xhat = np.zeros((4, 1), dtype=float)
        self.Pk = self.P_ss.copy()

        # history
        self.xhat_hist = None
        self.innov_hist = None
        self.Kg_norm_hist = None  # gain magnitude (diagnostic)

    def reset(self) -> None:
        self.xhat = np.zeros((4, 1), dtype=float)
        self.Pk = self.P_ss.copy()
        self.xhat_hist = None
        self.innov_hist = None
        self.Kg_norm_hist = None

    def setup_history(self, tn: int) -> None:
        self.xhat_hist = np.zeros((4, tn), dtype=float)
        self.innov_hist = np.zeros(tn, dtype=float)
        self.Kg_norm_hist = np.zeros(tn, dtype=float)

    def set_xhat0(self, xhat0: np.ndarray) -> None:
        xhat0 = np.asarray(xhat0, dtype=float).reshape(4, 1)
        self.xhat = xhat0

    def compute(self, *, x: np.ndarray, xdot: np.ndarray, y: float, dt: float, t: float) -> float:
        # Control uses current estimate
        u = float(-(self.K @ self.xhat).item())
        return u

    def kalman_update_with_measurement(self, u: float, y_meas: float, i: int) -> None:
        """
        One-step discrete KF update using measurement y_{k+1}.
        """
        # Predict
        xhat_pred = self.Ad @ self.xhat + self.Bd * float(u)
        P_pred = self.Ad @ self.Pk @ self.Ad.T + self.W_kf

        # Update
        S = self.Cd @ P_pred @ self.Cd.T + self.V_kf
        Kg = P_pred @ self.Cd.T @ np.linalg.inv(S)
        innov = float(y_meas - (self.Cd @ xhat_pred).item())

        self.xhat = xhat_pred + Kg * innov
        self.Pk = (np.eye(4) - Kg @ self.Cd) @ P_pred
        self.Pk = 0.5 * (self.Pk + self.Pk.T)

        # log
        if self.xhat_hist is not None:
            self.xhat_hist[:, i] = self.xhat.reshape(-1)
            self.innov_hist[i] = innov
            self.Kg_norm_hist[i] = float(np.linalg.norm(Kg))

    def export(self) -> Dict[str, Any]:
        return {
            "xhat": self.xhat_hist,
            "innov": self.innov_hist,
            "kalman_gain_norm": self.Kg_norm_hist,
            "K_lqr": self.K,
            "P_lqr": self.P_lqr,
            "P_kf_ss": self.P_ss,
            "L_kf_ss": self.L_ss,
        }


# ============================================================
#  Simulator
# ============================================================

@dataclass
class SimConfig:
    dt: float
    tn: int
    u_limit: Optional[float] = None
    meas_noise_std: float = 0.0  # measurement noise added to y (optional)


class StateSpaceSimulator:
    """
    Core simulator (no plotting). Returns a dict with full time histories for external plotting.
    """
    def __init__(self, system_config: Dict[str, float], noise_term: Dict[str, np.ndarray], sim_cfg: SimConfig):
        self.system_config = system_config
        self.noise_term = noise_term
        self.sim_cfg = sim_cfg

        self.A, self.B, self.C = assemble_matrices(system_config)
        self.C_row = self.C.reshape(1, -1)

    def _sat(self, u: float) -> float:
        lim = self.sim_cfg.u_limit
        if lim is None:
            return float(u)
        return float(np.clip(u, -lim, lim))

    def run(self, controller: ControllerBase, x0: Optional[np.ndarray] = None,
            xhat0: Optional[np.ndarray] = None) -> Dict[str, Any]:
        dt = self.sim_cfg.dt
        tn = self.sim_cfg.tn

        # histories
        t_axis = np.arange(tn) * dt
        X = np.zeros((4, tn), dtype=float)
        Xd = np.zeros((4, tn), dtype=float)
        Y = np.zeros((tn,), dtype=float)
        U = np.zeros((tn,), dtype=float)
        Fhist = np.zeros((4, tn), dtype=float)

        if x0 is not None:
            X[:, 0] = np.asarray(x0, dtype=float).reshape(4, )

        controller.reset()

        # controller-specific history init
        if hasattr(controller, "setup_history"):
            # FixedPIDController / LQGController / 你自定义 DRL-PID 都可以实现这个方法
            try:
                controller.setup_history(tn)
            except TypeError:
                pass

        if hasattr(controller, "set_xhat0") and xhat0 is not None:
            controller.set_xhat0(xhat0)

        # 关键：u_old = 上一步 u。与 env 一致：初始 u(-1)=0
        u_old = 0.0

        for i in range(tn - 1):
            t = t_axis[i]
            X_col = X[:, i].reshape(4, 1)

            # --- 1) disturbance ---
            F1 = compute_disturbance(self.noise_term, i, tn)
            F2 = compute_disturbance(self.noise_term, i + 1, tn)
            Fhist[:, i] = F1.reshape(-1)

            # --- 2) y_now at step i ---
            y_now = float((self.C_row @ X_col).item())
            if self.sim_cfg.meas_noise_std > 0:
                y_now = y_now + np.random.randn() * self.sim_cfg.meas_noise_std
            Y[i] = y_now

            # --- 3) Xd_col using u_old (env-consistent) ---
            Xd_col = self.A @ X_col + self.B * u_old + F1
            Xd[:, i] = Xd_col.reshape(-1)

            # --- 4) controller computes u_new using current signals ---
            # 注意：这里把 u_prev 的语义交给 controller（DRL-PID 会用它构造 state）
            u_new_cmd = controller.compute(
                x=X[:, i].copy(),
                xdot=Xd[:, i].copy(),  # 这是基于 u_old 的 xdot，与 env 的 ed 计算一致
                y=y_now,
                dt=dt,
                t=t
            )
            u_new = self._sat(u_new_cmd)
            U[i] = u_new

            # 控制器日志（支持 FixedPIDController / 你自定义 DRL-PID）
            if hasattr(controller, "log_step"):
                controller.log_step(i)

            # --- 5) RK4: k1 uses u_old; k2/k3/k4 use u_new (env-consistent) ---
            k1_col = dt * Xd_col

            k2_col = dt * (self.A @ (X_col + k1_col / 2.0) +
                           self.B * u_new + (F1 + F2) / 2.0)

            k3_col = dt * (self.A @ (X_col + k2_col / 2.0) +
                           self.B * u_new + (F1 + F2) / 2.0)

            k4_col = dt * (self.A @ (X_col + k3_col) +
                           self.B * u_new + F2)

            X_update_col = X_col + (k1_col + 2 * k2_col + 2 * k3_col + k4_col) / 6.0
            X[:, i + 1] = X_update_col.reshape(-1)

            # --- 6) y_next at step i+1 ---
            y_next = float((self.C_row @ X_update_col).item())
            if self.sim_cfg.meas_noise_std > 0:
                y_next = y_next + np.random.randn() * self.sim_cfg.meas_noise_std
            Y[i + 1] = y_next

            # LQG estimator update uses y_{i+1} (与你之前的实现一致)
            if hasattr(controller, "kalman_update_with_measurement"):
                controller.kalman_update_with_measurement(u=u_new, y_meas=y_next, i=i + 1)

            # --- shift for next step ---
            u_old = u_new

        # last bookkeeping
        Fhist[:, -1] = compute_disturbance(self.noise_term, tn - 1, tn).reshape(-1)
        U[-1] = u_old
        x_last = X[:, -1].reshape(4, 1)
        Xd[:, -1] = (self.A @ x_last + self.B * u_old + compute_disturbance(self.noise_term, tn - 1, tn)).reshape(-1)

        results = {
            "t": t_axis,
            "A": self.A, "B": self.B, "C": self.C,
            "X": X, "Xd": Xd, "Y": Y, "U": U, "F": Fhist,
            "meta": {
                "dt": dt, "tn": tn,
                "u_limit": self.sim_cfg.u_limit,
                "meas_noise_std": self.sim_cfg.meas_noise_std,
            },
            "controller": controller.export()
        }
        return results


# ============================================================
#  Controller factories
# ============================================================

def make_fixed_pid_controller(system_config: Dict[str, float], kp: float, ki: float, kd: float) -> FixedPIDController:
    _, _, C = assemble_matrices(system_config)
    return FixedPIDController(PIDGains(kp=kp, ki=ki, kd=kd), C=C)


def make_lqg_controller(system_config: Dict[str, float], dt: float,
                        Q_lqr: Optional[np.ndarray] = None,
                        R_lqr: Optional[np.ndarray] = None,
                        W_kf: Optional[np.ndarray] = None,
                        V_kf: Optional[np.ndarray] = None) -> LQGController:
    A, B, C = assemble_matrices(system_config)

    # LQR weights (hyperparameters)
    if Q_lqr is None:
        Q_lqr = np.diag([1, 1, 10, 10]).astype(float)
    if R_lqr is None:
        R_lqr = np.array([[1e-6]], dtype=float)

    # KF covariances (hyperparameters)
    if W_kf is None:
        # default: allow more uncertainty in velocities than positions
        W_kf = np.diag([1e-6, 1e-6, 1e-3, 1e-3]).astype(float)
    if V_kf is None:
        V_kf = np.array([[1e-4]], dtype=float)

    return LQGController(A=A, B=B, C=C, dt=dt, Q_lqr=Q_lqr, R_lqr=R_lqr, W_kf=W_kf, V_kf=V_kf)
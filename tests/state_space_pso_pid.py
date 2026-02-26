import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing as mp

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# =========================================================
# State-space simulator (STRICTLY follows your implementation)
# X'(t) = A X(t) + B u(t) + F(t)
# Y(t)  = C X(t)
#
# PID:
# e  = C X
# ei = ∫ e dt  (discrete accumulation)
# ed = C Xd    where Xd = A X + B u + F
# u  = kp*e - ki*ei - kd*ed
#
# RK4 integration EXACTLY as your structure
# =========================================================
class StateSpace:
    def __init__(self, config, noise_term, dt=0.01, tn=10000, dx=0.005, kp=0.0, ki=0.0, kd=0.0):
        self.config = config
        self.noise_term = noise_term
        self.dt = dt
        self.tn = tn
        self.dx = dx
        self.Reference = np.zeros(tn)
        self.current_step = 0

        # PID parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # PID states
        self.e = 0
        self.ei = 0
        self.ed = 0

        # State/output arrays
        self.X = np.zeros((4, tn))
        self.Xd = np.zeros((4, tn))   # store Xdot at each step (last step filled at end)
        self.Y = np.zeros(tn)
        self.u = 0

        # Noise
        self.F1 = self.noise_term["F1"]
        self.F2 = self.noise_term["F2"]

        # Matrices
        self.A, self.B, self.C = self.assemble_mat()

        # Optional external controller (not used in PSO)
        self.external_controller_callback = None

        # Histories
        self.kp_history = np.zeros(tn)
        self.ki_history = np.zeros(tn)
        self.kd_history = np.zeros(tn)
        self.u_history = np.zeros(tn)

    def set_external_controller(self, controller_callback):
        self.external_controller_callback = controller_callback

    def assemble_mat(self):
        w1 = self.config["w1"]
        w2 = self.config["w2"]
        z1 = self.config["z1"]
        z2 = self.config["z2"]
        B1 = self.config["B1"]
        B2 = self.config["B2"]
        C1 = self.config["C1"]
        C2 = self.config["C2"]

        A = np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [-w1**2, 0, -2 * z1 * w1, 0],
                [0, -w2**2, 0, -2 * z2 * w2],
            ],
            dtype=float,
        )
        B = np.array([[0], [0], [B1], [B2]], dtype=float)
        C = np.array([C1, C2, 0, 0], dtype=float)
        return A, B, C

    def compute_noise(self, idx):
        if idx < self.tn:
            F = np.array([[0], [0], [self.F1[idx, 1]], [self.F2[idx, 1]]], dtype=float)
        else:
            F = np.zeros((4, 1), dtype=float)
        return F

    def solve(self):
        # Ensure B shape (4,1)
        if self.B.ndim == 1 or self.B.shape != (4, 1):
            self.B = self.B.reshape(4, 1)

        for i in range(self.tn - 1):
            self.current_step = i

            # record PID params
            self.kp_history[i] = self.kp
            self.ki_history[i] = self.ki
            self.kd_history[i] = self.kd

            X_col = self.X[:, i].reshape(4, 1)

            # noise
            F1 = self.compute_noise(i)
            F2 = self.compute_noise(i + 1)

            if F1.ndim == 1 or F1.shape != (4, 1):
                F1 = F1.reshape(4, 1)
            if F2.ndim == 1 or F2.shape != (4, 1):
                F2 = F2.reshape(4, 1)

            # Xdot at time i
            Xd_col = self.A @ X_col + self.B * self.u + F1
            self.Xd[:, i] = Xd_col.reshape(-1)

            # PID
            self.e = (self.C @ X_col)
            self.ei += (self.e * self.dt)
            self.ed = (self.C @ Xd_col)

            self.u = self.kp * self.e - self.ki * self.ei - self.kd * self.ed
            self.u_history[i] = (self.u).item()

            # external controller (kept)
            if self.external_controller_callback is not None:
                current_time = i * self.dt
                current_output = self.Y[i]
                new_kp, new_ki, new_kd = self.external_controller_callback(current_output, current_time)
                self.kp = new_kp
                self.ki = new_ki
                self.kd = new_kd

            # RK4
            k1_col = self.dt * Xd_col
            k2_col = self.dt * (self.A @ (X_col + k1_col / 2) + self.B * self.u + (F1 + F2) / 2)
            k3_col = self.dt * (self.A @ (X_col + k2_col / 2) + self.B * self.u + (F1 + F2) / 2)
            k4_col = self.dt * (self.A @ (X_col + k3_col) + self.B * self.u + F2)

            X_update_col = X_col + (k1_col + 2 * k2_col + 2 * k3_col + k4_col) / 6.0
            self.X[:, i + 1] = X_update_col.reshape(-1)

            Y_next = self.C @ X_update_col
            self.Y[i + 1] = Y_next.item()

        # last-step histories
        self.kp_history[-1] = self.kp
        self.ki_history[-1] = self.ki
        self.kd_history[-1] = self.kd
        self.u_history[-1] = self.u_history[-2]

        # Fill last Xdot strictly: Xdot = A X + B u + F
        X_last = self.X[:, -1].reshape(4, 1)
        F_last = self.compute_noise(self.tn - 1)
        Xd_last = self.A @ X_last + self.B * self.u_history[-1] + F_last
        self.Xd[:, -1] = Xd_last.reshape(-1)

    def plot_time_domain_response(self, save_path="../../results/state_response.png"):
        N = self.X.shape[1]
        time_vector = np.arange(N) * self.dt
        z = -2 * self.X[0, :] + 2 * self.X[1, :]

        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        ax1.plot(time_vector, z, linewidth=1.5, label=r"$z = -2X_0 + 2X_1$")
        ax1.set_ylabel("位移 (m)", fontsize=12)
        ax1.set_title("端部动力学响应 (真实值)", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle=":", alpha=0.7)

        ax2.plot(time_vector, self.Y.flatten(), linewidth=1.5, label="Sensor Output")
        ax2.set_ylabel("电压 (V)", fontsize=12)
        ax2.set_title("传感器实测电压信号", fontsize=14, fontweight="bold")
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle=":", alpha=0.7)

        ax3.plot(time_vector, self.u_history.flatten(), linewidth=1.5, label="Control Input $u(t)$")
        ax3.set_xlabel("时间 (s)", fontsize=12)
        ax3.set_ylabel("u", fontsize=12)
        ax3.set_title("控制输入时间历程", fontsize=14, fontweight="bold")
        ax3.legend(loc="upper right")
        ax3.grid(True, linestyle=":", alpha=0.7)

        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


# =========================================================
# Multiprocessing helpers (global vars in workers)
# =========================================================
_G_CONFIG_SYSTEM = None
_G_NOISE_LIST = None
_G_DT = None
_G_TN = None
_G_YDOT_W = None
_G_FAIL = None
_G_UCLIP = None

def _worker_init(config_system, noise_data_list, dt, tn, ydot_weight, fail_penalty, u_clip):
    global _G_CONFIG_SYSTEM, _G_NOISE_LIST, _G_DT, _G_TN, _G_YDOT_W, _G_FAIL, _G_UCLIP
    _G_CONFIG_SYSTEM = config_system
    _G_NOISE_LIST = noise_data_list
    _G_DT = float(dt)
    _G_TN = int(tn)
    _G_YDOT_W = float(ydot_weight)
    _G_FAIL = float(fail_penalty)
    _G_UCLIP = u_clip

def _eval_one_particle(pid_params):
    """
    Evaluate one particle (kp, ki, kd):
    objective = mean over 5 envs of ∫ (y^2 + 10*y'^2) dt
    where y' = C Xdot = C Xd (STRICTLY from state-space)
    """
    kp, ki, kd = pid_params
    try:
        vals = []
        for noise_data in _G_NOISE_LIST:
            ss = StateSpace(
                _G_CONFIG_SYSTEM,
                noise_data,
                dt=_G_DT,
                tn=_G_TN,
                kp=float(kp),
                ki=float(ki),
                kd=float(kd),
            )
            ss.solve()

            # guards
            if (not np.isfinite(ss.X).all()) or (not np.isfinite(ss.Xd).all()) or (not np.isfinite(ss.Y).all()):
                return _G_FAIL
            if (not np.isfinite(ss.u_history).all()):
                return _G_FAIL

            if _G_UCLIP is not None:
                umin, umax = _G_UCLIP
                if np.any(ss.u_history < umin) or np.any(ss.u_history > umax):
                    return _G_FAIL

            C = ss.C.reshape(1, 4)
            y = (C @ ss.X).reshape(-1)
            ydot = (C @ ss.Xd).reshape(-1)

            if (not np.isfinite(y).all()) or (not np.isfinite(ydot).all()):
                return _G_FAIL

            J_env = _G_DT * float(np.sum(y * y + _G_YDOT_W * (ydot * ydot)))
            if (not np.isfinite(J_env)) or (J_env >= _G_FAIL):
                return _G_FAIL

            vals.append(J_env)

        return float(np.mean(vals))
    except Exception:
        return _G_FAIL


# =========================================================
# PSO (particle evaluation parallelized by 14 workers)
# =========================================================
class PSO_PID:
    def __init__(
        self,
        config_system,
        noise_data_list,   # length=5
        dt,
        tn,
        kp_bounds=(0.0, 500.0),
        ki_bounds=(0.0, 500.0),
        kd_bounds=(0.0, 200.0),
        n_particles=30,
        n_iters=60,
        w=0.72,
        c1=1.49,
        c2=1.49,
        seed=42,
        ydot_weight=10.0,
        fail_penalty=1e18,
        u_clip=None,
        n_workers=14,      # <<< NEW: 14 cores parallel
        mp_start_method=None,  # None -> choose automatically; or "spawn"/"fork"
    ):
        self.config_system = config_system
        self.noise_data_list = list(noise_data_list)
        if len(self.noise_data_list) != 5:
            raise ValueError(f"noise_data_list must have length 5, got {len(self.noise_data_list)}")

        self.dt = float(dt)
        self.tn = int(tn)

        self.bounds = np.array([kp_bounds, ki_bounds, kd_bounds], dtype=float)  # (3,2)
        self.n_particles = int(n_particles)
        self.n_iters = int(n_iters)

        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.rng = np.random.default_rng(seed)

        self.ydot_weight = float(ydot_weight)
        self.fail_penalty = float(fail_penalty)
        self.u_clip = u_clip

        self.n_workers = int(n_workers)
        self.mp_start_method = mp_start_method  # allow override

        # init positions & velocities
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        self.pos = self.rng.uniform(low, high, size=(self.n_particles, 3))
        vel_scale = 0.1 * (high - low)
        self.vel = self.rng.uniform(-vel_scale, vel_scale, size=(self.n_particles, 3))

        # bests
        self.pbest_pos = self.pos.copy()
        self.pbest_val = np.full((self.n_particles,), np.inf, dtype=float)
        self.gbest_pos = None
        self.gbest_val = np.inf

        self.best_history = []

    def _clip_pos(self, x):
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        return np.minimum(np.maximum(x, low), high)

    def run(self, verbose=True):
        # choose multiprocessing context
        if self.mp_start_method is None:
            # "fork" is fastest on Linux, but "spawn" is safer cross-platform.
            # If you're on Linux and want speed, set mp_start_method="fork".
            ctx = mp.get_context("spawn")
        else:
            ctx = mp.get_context(self.mp_start_method)

        with ctx.Pool(
            processes=self.n_workers,
            initializer=_worker_init,
            initargs=(
                self.config_system,
                self.noise_data_list,
                self.dt,
                self.tn,
                self.ydot_weight,
                self.fail_penalty,
                self.u_clip,
            ),
        ) as pool:

            for it in range(self.n_iters):
                # parallel evaluation of all particles
                vals = pool.map(_eval_one_particle, [tuple(p) for p in self.pos], chunksize=1)
                vals = np.asarray(vals, dtype=float)

                # update personal and global best
                improved = vals < self.pbest_val
                self.pbest_val[improved] = vals[improved]
                self.pbest_pos[improved] = self.pos[improved]

                idx = int(np.argmin(self.pbest_val))
                if self.pbest_val[idx] < self.gbest_val:
                    self.gbest_val = float(self.pbest_val[idx])
                    self.gbest_pos = self.pbest_pos[idx].copy()

                self.best_history.append(self.gbest_val)

                # PSO update
                r1 = self.rng.random((self.n_particles, 3))
                r2 = self.rng.random((self.n_particles, 3))
                cognitive = self.c1 * r1 * (self.pbest_pos - self.pos)
                social = self.c2 * r2 * (self.gbest_pos - self.pos)

                self.vel = self.w * self.vel + cognitive + social
                self.pos = self._clip_pos(self.pos + self.vel)

                if verbose:
                    bk, bi, bd = self.gbest_pos
                    print(
                        f"[PSO-14C] iter={it+1:03d}/{self.n_iters}, best J(avg5)={self.gbest_val:.6e}, "
                        f"kp={bk:.6g}, ki={bi:.6g}, kd={bd:.6g}"
                    )

        return self.gbest_pos.copy(), float(self.gbest_val)

    def plot_convergence(self, save_path="../../results/pso_convergence.png"):
        plt.figure(figsize=(9, 4))
        plt.plot(np.arange(1, len(self.best_history) + 1), self.best_history, linewidth=1.5)
        plt.xlabel("Iteration")
        plt.ylabel("Best objective J (avg over 5 envs)")
        plt.title("PSO Convergence")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


# =========================================================
# Main: build 5 disturbance environments -> PSO(14-core) -> validate
# =========================================================
if __name__ == "__main__":
    from config.config import Config
    from config.config_loader import load_mat
    from src.utils.utils import BeamDisturbanceProjector, create_noise_data

    # 1) base config
    config = Config()
    tn = int(config.EPISODE_LENGTH)
    dt = float(config.DT)

    # 2) load mt data
    try:
        mt = load_mat()
    except Exception as e:
        print(f"Warning: mt_data load failed: {e}")
        mt = None

    # 3) projector coeffs
    proj = BeamDisturbanceProjector()
    projector_coeffs = proj.get_static_coeffs()

    # 4) build 5 environments (EDIT to match your create_noise_data options)
    env_options = ["impact", "thermal", "jitter", "maneuver", "mixed"]
    env_seeds = [101, 102, 103, 104, 105]

    noise_data_list = []
    for opt, sd in zip(env_options, env_seeds):
        np.random.seed(sd)  # reproducible if create_noise_data uses np.random
        nd = create_noise_data(
            tn,
            option=opt,
            system_config=config.SYSTEM_CONFIG,
            mt_data=mt,
            projector_data=projector_coeffs,
        )
        noise_data_list.append(nd)

    # 5) PSO with 14 workers
    pso = PSO_PID(
        config_system=config.SYSTEM_CONFIG,
        noise_data_list=noise_data_list,
        dt=dt,
        tn=tn,
        kp_bounds=(0.0, 180.0),
        ki_bounds=(0.0, 100.0),
        kd_bounds=(0.0, 30.0),
        n_particles=30,
        n_iters=60,
        w=0.72,
        c1=1.49,
        c2=1.49,
        seed=123,
        ydot_weight=0.1,
        fail_penalty=1e18,
        u_clip=None,
        n_workers=14,
        mp_start_method=None,  # None->spawn; if Linux and want speed: "fork"
    )

    print("Running PSO (14-core parallel): objective = mean_{5 env} ∫(y^2 + 10 y'^2) dt ...")
    best_pid, best_J = pso.run(verbose=True)
    best_kp, best_ki, best_kd = best_pid

    print("\n========== PSO RESULT ==========")
    print(f"Best J(avg5) = {best_J:.6e}")
    print(f"Best kp      = {best_kp:.10g}")
    print(f"Best ki      = {best_ki:.10g}")
    print(f"Best kd      = {best_kd:.10g}")
    print("================================\n")

    pso.plot_convergence(save_path="../../results/pso_convergence.png")

    # 6) Validate best PID on each env and print per-env + mean
    per_env = []
    for opt, nd in zip(env_options, noise_data_list):
        ss = StateSpace(config.SYSTEM_CONFIG, nd, dt=dt, tn=tn, kp=best_kp, ki=best_ki, kd=best_kd)
        ss.solve()

        C = ss.C.reshape(1, 4)
        y = (C @ ss.X).reshape(-1)
        ydot = (C @ ss.Xd).reshape(-1)
        J = dt * float(np.sum(y * y + 10.0 * (ydot * ydot)))
        per_env.append(J)
        print(f"[VALID] option={opt:<8s}  J_env={J:.6e}")

    print(f"[VALID] mean over 5 envs  J_mean={float(np.mean(per_env)):.6e}")

    # 7) Plot one representative env (e.g., impact) with best PID
    ss_plot = StateSpace(config.SYSTEM_CONFIG, noise_data_list[0], dt=dt, tn=tn, kp=best_kp, ki=best_ki, kd=best_kd)
    ss_plot.solve()
    ss_plot.plot_time_domain_response(save_path="../../results/state_response_best_pid_env0.png")
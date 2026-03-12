import numpy as np


class PIDControlEnvironment:
    """StateSpace-based RL environment (SISO + PID gains as actions)."""

    def __init__(self, config):
        self.config = config
        self.state_space = None
        self.current_step = 0
        self.episode_length = config.EPISODE_LENGTH
        self.dt = config.DT

        # state = [u_prev, e, ei, ed, t_norm] (keep 5D to minimize changes)
        self.state = np.zeros(config.STATE_DIM, dtype=float)

        # Action smoothness penalty
        self.last_pid_params = np.array(
            [config.KP_RANGE[0], config.KI_RANGE[0], config.KD_RANGE[0]],
            dtype=float
        )

        # Reward accumulators
        self.cumulative_output_squared = 0.0
        self.cumulative_control_effort = 0.0

        # Cache u_prev (last applied input)
        self.u_prev = 0.0

    def set_state_space(self, state_space):
        """Attach a StateSpace instance."""
        self.state_space = state_space

        # Initialize PID history arrays (kept consistent with legacy behavior)
        self.state_space.kp = np.ones(self.episode_length) * self.config.KP_RANGE[0]
        self.state_space.ki = np.ones(self.episode_length) * self.config.KI_RANGE[0]
        self.state_space.kd = np.ones(self.episode_length) * self.config.KD_RANGE[0]

        # Ensure Y exists
        if not hasattr(self.state_space, "Y") or self.state_space.Y is None:
            self.state_space.Y = np.zeros(self.episode_length, dtype=float)

    def reset(self, initial_state=None):
        """Reset environment (aligned with StateSpace init logic)."""
        if self.state_space is None:
            raise ValueError("StateSpace instance not set.")

        # 1) Reset physical states
        self.state_space.X = np.zeros((4, self.episode_length), dtype=float)
        self.state_space.Xd = np.zeros((4, self.episode_length), dtype=float)
        self.state_space.Y = np.zeros(self.episode_length, dtype=float)

        if initial_state is not None:
            self.state_space.X[:, 0] = np.asarray(initial_state, dtype=float).reshape(-1)

        # 2) Reset PID internal states
        self.state_space.e = 0.0
        self.state_space.ei = 0.0
        self.state_space.ed = 0.0
        self.state_space.u = 0.0  # u_(-1)=0

        # 3) Reset counters and caches
        self.current_step = 0
        self.u_prev = 0.0
        self.cumulative_output_squared = 0.0
        self.cumulative_control_effort = 0.0

        # 4) Reset last PID params
        self.last_pid_params = np.array(
            [self.config.KP_RANGE[0], self.config.KI_RANGE[0], self.config.KD_RANGE[0]],
            dtype=float
        )

        # 5) Initial observation (t=0)
        # No control applied yet; e/ei/ed/u are all zero; t_norm=0
        self._update_rl_state()
        return self.state.copy()

    def _update_rl_state(self):
        """
        Build RL state vector aligned to time index i:
        s_i = [u_{i-1}, e_i, ei_i, ed_i, i/T]
        """
        i = self.current_step
        if i >= self.episode_length:
            i = self.episode_length - 1

        u_prev = float(self.u_prev)
        e = float(self.state_space.e)
        ei = float(self.state_space.ei)
        ed = float(self.state_space.ed)
        t_norm = float(i / self.episode_length)

        self.state = np.array([u_prev, e, ei, ed, t_norm], dtype=float)

    def _calculate_reward(self, output, control_force, pid_params):
        """Reward = - (output^2 + control^2 + param_change_penalty + divergence_penalty)."""
        output_penalty = self.config.OUTPUT_WEIGHT * (output ** 2)

        param_diff = np.sum((pid_params - self.last_pid_params) ** 2)
        param_penalty = self.config.PARAM_PENALTY * param_diff

        stability_penalty = 0.0
        if abs(output) > 500.0:
            stability_penalty = 1e6

        self.last_pid_params = pid_params.copy()

        self.cumulative_output_squared += output ** 2
        self.cumulative_control_effort += control_force ** 2

        reward = - (output_penalty + param_penalty + stability_penalty)
        return float(reward)

    def step(self, action):
        """
        Causal order (avoid algebraic loop):
        - Use u_prev (= previous u) to compute Xd_col, then ed
        - Use current (kp,ki,kd) to compute u_new
        - RK4: k1 uses u_prev; k2/k3/k4 use u_new
        """
        if self.state_space is None:
            raise ValueError("StateSpace instance not set.")

        i = self.current_step
        if i >= self.episode_length - 1:
            return self.state.copy(), 0.0, True, {}

        # --- 0) Denormalize action -> PID gains ---
        pid_params = np.asarray(self.config.denormalize_action(action), dtype=float)
        kp, ki, kd = float(pid_params[0]), float(pid_params[1]), float(pid_params[2])

        # Log history
        self.state_space.kp[i] = kp
        self.state_space.ki[i] = ki
        self.state_space.kd[i] = kd

        # Current state
        X_col = self.state_space.X[:, i].reshape(4, 1)

        # Previous input u_{i-1}
        u_old = float(self.state_space.u)
        self.u_prev = u_old  # cached for state

        # --- 1) Disturbance ---
        F1 = self.state_space.compute_noise(i)
        F2 = self.state_space.compute_noise(i + 1)
        if F1.ndim == 1:
            F1 = F1.reshape(4, 1)
        if F2.ndim == 1:
            F2 = F2.reshape(4, 1)

        # --- 2) Current output y_now at step i, then store Y[i] ---
        y_now = (self.state_space.C @ X_col).item()
        self.state_space.Y[i] = float(y_now)

        # --- 3) Xdot using u_old, then ed at step i ---
        Xd_col = self.state_space.A @ X_col + self.state_space.B * u_old + F1
        self.state_space.Xd[:, i] = Xd_col.reshape(-1)

        # Error definition (ref = 0): e = y
        e = float(y_now)
        self.state_space.e = e

        # Integral
        self.state_space.ei += e * self.dt
        ei = float(self.state_space.ei)

        # Derivative (based on u_old)
        ed = (self.state_space.C @ Xd_col).item()
        self.state_space.ed = float(ed)

        # --- 4) New control input u_new at step i ---
        # Sign convention: +P, -I, -D
        u_new = kp * e - ki * ei - kd * ed
        self.state_space.u = float(u_new)

        # --- 5) RK4: k1 uses u_old; k2/k3/k4 use u_new ---
        k1_col = self.dt * Xd_col

        k2_col = self.dt * (self.state_space.A @ (X_col + k1_col / 2) +
                            self.state_space.B * u_new + (F1 + F2) / 2)

        k3_col = self.dt * (self.state_space.A @ (X_col + k2_col / 2) +
                            self.state_space.B * u_new + (F1 + F2) / 2)

        k4_col = self.dt * (self.state_space.A @ (X_col + k3_col) +
                            self.state_space.B * u_new + F2)

        X_update_col = X_col + (k1_col + 2 * k2_col + 2 * k3_col + k4_col) / 6.0
        self.state_space.X[:, i + 1] = X_update_col.reshape(-1)

        # --- 6) Update output Y[i+1] ---
        Y_next = (self.state_space.C @ X_update_col).item()
        self.state_space.Y[i + 1] = float(Y_next)

        # --- 7) Reward: use y_now and u_new (time-consistent) ---
        reward = self._calculate_reward(output=float(y_now), control_force=float(u_new), pid_params=pid_params)

        # --- 8) Advance step and update RL state ---
        self.current_step += 1
        done = self.current_step >= self.episode_length - 1

        # Note: e/ei/ed are from step i, while current_step is i+1.
        # This yields a consistent post-step observation.
        self._update_rl_state()

        info = {
            'output': float(y_now),
            'control_force': float(u_new),
            'kp': kp, 'ki': ki, 'kd': kd
        }

        return self.state.copy(), reward, done, info

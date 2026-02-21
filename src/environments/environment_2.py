import numpy as np


class PIDControlEnvironment:
    """基于StateSpace的强化学习环境封装（SISO + PID参数作为动作）"""

    def __init__(self, config):
        self.config = config
        self.state_space = None
        self.current_step = 0
        self.episode_length = config.EPISODE_LENGTH
        self.dt = config.DT

        # state = [u_prev, e, ei, ed, t_norm]  (保持5维，最小改动)
        self.state = np.zeros(config.STATE_DIM, dtype=float)

        # 动作平滑惩罚用
        self.last_pid_params = np.array(
            [config.KP_RANGE[0], config.KI_RANGE[0], config.KD_RANGE[0]],
            dtype=float
        )

        # 用于奖励累计
        self.cumulative_output_squared = 0.0
        self.cumulative_control_effort = 0.0

        # 缓存 u_prev（上一拍输入）
        self.u_prev = 0.0

    def set_state_space(self, state_space):
        """设置 StateSpace 实例"""
        self.state_space = state_space

        # 初始化参数记录（与你旧版本一致）
        self.state_space.kp = np.ones(self.episode_length) * self.config.KP_RANGE[0]
        self.state_space.ki = np.ones(self.episode_length) * self.config.KI_RANGE[0]
        self.state_space.kd = np.ones(self.episode_length) * self.config.KD_RANGE[0]

        # 确保 Y 有空间
        if not hasattr(self.state_space, "Y") or self.state_space.Y is None:
            self.state_space.Y = np.zeros(self.episode_length, dtype=float)

    def reset(self, initial_state=None):
        """重置环境 - 与 StateSpace 初始状态逻辑一致"""
        if self.state_space is None:
            raise ValueError("StateSpace instance not set.")

        # 1) 重置物理状态
        self.state_space.X = np.zeros((4, self.episode_length), dtype=float)
        self.state_space.Xd = np.zeros((4, self.episode_length), dtype=float)
        self.state_space.Y = np.zeros(self.episode_length, dtype=float)

        if initial_state is not None:
            self.state_space.X[:, 0] = np.asarray(initial_state, dtype=float).reshape(-1)

        # 2) 重置 PID 内部状态
        self.state_space.e = 0.0
        self.state_space.ei = 0.0
        self.state_space.ed = 0.0
        self.state_space.u = 0.0  # u_(-1)=0

        # 3) 重置缓存/计数
        self.current_step = 0
        self.u_prev = 0.0
        self.cumulative_output_squared = 0.0
        self.cumulative_control_effort = 0.0

        # 4) reset last pid params
        self.last_pid_params = np.array(
            [self.config.KP_RANGE[0], self.config.KI_RANGE[0], self.config.KD_RANGE[0]],
            dtype=float
        )

        # 5) 生成初始观测（t=0）
        # 注意：此时还没执行控制，e/ei/ed/u 都是0，t_norm=0
        self._update_rl_state()
        return self.state.copy()

    def _update_rl_state(self):
        """
        返回给RL的状态向量（与 step 的时序一致）：
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
        """奖励 = - (输出平方 + 控制平方 + 参数变化惩罚 + 发散惩罚)"""
        output_penalty = self.config.OUTPUT_WEIGHT * (output ** 2)
        control_penalty = self.config.CONTROL_WEIGHT * (control_force ** 2)

        param_diff = np.sum((pid_params - self.last_pid_params) ** 2)
        param_penalty = self.config.PARAM_PENALTY * param_diff

        stability_penalty = 0.0
        if abs(output) > 500.0:
            stability_penalty = 1e6

        self.last_pid_params = pid_params.copy()

        self.cumulative_output_squared += output ** 2
        self.cumulative_control_effort += control_force ** 2

        reward = - (output_penalty + control_penalty + param_penalty + stability_penalty)
        return float(reward)

    def step(self, action):
        """
        与旧逻辑一致的因果顺序（避免代数环）：
        - 用 u_prev (=上一步 self.state_space.u) 先算 Xd_col，再算 ed
        - 再用当前 (kp,ki,kd) 计算 u_new
        - RK4：k1 用 u_prev，k2/k3/k4 用 u_new
        """
        if self.state_space is None:
            raise ValueError("StateSpace instance not set.")

        i = self.current_step
        if i >= self.episode_length - 1:
            return self.state.copy(), 0.0, True, {}

        # --- 0) 动作反归一化 -> PID 参数 ---
        pid_params = np.asarray(self.config.denormalize_action(action), dtype=float)
        kp, ki, kd = float(pid_params[0]), float(pid_params[1]), float(pid_params[2])

        # 记录历史
        self.state_space.kp[i] = kp
        self.state_space.ki[i] = ki
        self.state_space.kd[i] = kd

        # 当前状态
        X_col = self.state_space.X[:, i].reshape(4, 1)

        # 上一步输入（u_{i-1}）
        u_old = float(self.state_space.u)
        self.u_prev = u_old  # 缓存给 state 使用

        # --- 1) 噪声 ---
        F1 = self.state_space.compute_noise(i)
        F2 = self.state_space.compute_noise(i + 1)
        if F1.ndim == 1:
            F1 = F1.reshape(4, 1)
        if F2.ndim == 1:
            F2 = F2.reshape(4, 1)

        # --- 2) 当前输出 y_now（时刻 i），并写入 Y[i] ---
        y_now = (self.state_space.C @ X_col).item()
        self.state_space.Y[i] = float(y_now)

        # --- 3) 用 u_old 计算 Xd_col，并由此计算 ed（时刻 i） ---
        Xd_col = self.state_space.A @ X_col + self.state_space.B * u_old + F1
        self.state_space.Xd[:, i] = Xd_col.reshape(-1)

        # 误差定义：按你旧版本 = 输出（ref=0）
        e = float(y_now)
        self.state_space.e = e

        # 积分
        self.state_space.ei += e * self.dt
        ei = float(self.state_space.ei)

        # 导数（注意：此处是基于 u_old 的导数代理）
        ed = (self.state_space.C @ Xd_col).item()
        self.state_space.ed = float(ed)

        # --- 4) 计算新输入 u_new（时刻 i 的控制） ---
        # 符号沿用你旧版本：正P，负I，负D
        u_new = kp * e - ki * ei - kd * ed
        self.state_space.u = float(u_new)

        # --- 5) RK4：k1 用 u_old，k2/k3/k4 用 u_new ---
        k1_col = self.dt * Xd_col

        k2_col = self.dt * (self.state_space.A @ (X_col + k1_col / 2) +
                            self.state_space.B * u_new + (F1 + F2) / 2)

        k3_col = self.dt * (self.state_space.A @ (X_col + k2_col / 2) +
                            self.state_space.B * u_new + (F1 + F2) / 2)

        k4_col = self.dt * (self.state_space.A @ (X_col + k3_col) +
                            self.state_space.B * u_new + F2)

        X_update_col = X_col + (k1_col + 2 * k2_col + 2 * k3_col + k4_col) / 6.0
        self.state_space.X[:, i + 1] = X_update_col.reshape(-1)

        # --- 6) 更新输出 Y[i+1] ---
        Y_next = (self.state_space.C @ X_update_col).item()
        self.state_space.Y[i + 1] = float(Y_next)

        # --- 7) reward：用同一拍 i 的 y_now 与 u_new（时序一致） ---
        reward = self._calculate_reward(output=float(y_now), control_force=float(u_new), pid_params=pid_params)

        # --- 8) 推进步计数并更新 state（对应下一次决策时刻 i+1 的 t_norm） ---
        self.current_step += 1
        done = self.current_step >= self.episode_length - 1

        # 注意：这里 state_space.e/ei/ed 仍是“时刻 i 的控制器内部量”
        # 但 current_step 已变为 i+1，所以 _update_rl_state 的 t_norm 是 i+1/T，
        # 其余量仍代表最新计算完的那一拍（i）。这是因果一致的“post-step observation”。
        self._update_rl_state()

        info = {
            'output': float(y_now),
            'control_force': float(u_new),
            'kp': kp, 'ki': ki, 'kd': kd
        }

        return self.state.copy(), reward, done, info
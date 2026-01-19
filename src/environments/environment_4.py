import numpy as np


class PIDControlEnvironment:
    """基于StateSpace的强化学习环境封装"""

    def __init__(self, config):
        self.config = config
        self.state_space = None
        self.current_step = 0
        self.episode_length = config.EPISODE_LENGTH
        self.dt = config.DT

        # 状态变量
        self.state = np.zeros(config.STATE_DIM)
        self.last_pid_params = np.array([config.KP_RANGE[0], config.KI_RANGE[0], config.KD_RANGE[0]])

        # 用于奖励计算
        self.cumulative_output_squared = 0
        self.cumulative_control_effort = 0

    def set_state_space(self, state_space):
        """设置StateSpace实例"""
        self.state_space = state_space

        # 设置初始PID参数
        self.state_space.kp = self.config.KP_RANGE[0]
        self.state_space.ki = self.config.KI_RANGE[0]
        self.state_space.kd = self.config.KD_RANGE[0]

    def reset(self, initial_state=None):
        """重置环境"""
        if self.state_space is None:
            raise ValueError("StateSpace instance not set. Call set_state_space first.")

        # 重置StateSpace
        if initial_state is not None:
            self.state_space.X[:, 0] = initial_state
        else:
            # 随机初始状态（自由振动训练）
            if np.random.random() < 0.5:
                # 初始位移
                self.state_space.X[:, 0] = [np.random.uniform(-1, 1), 0, 0, 0, 0, 0, 0, 0]
            else:
                # 初始速度
                self.state_space.X[:, 0] = [0, 0, 0, 0, np.random.uniform(-1, 1), 0, 0, 0]

        # 重置PID状态
        self.state_space.e = 0
        self.state_space.e_last = 0
        self.state_space.ei = 0
        self.state_space.ed = 0

        # 重置步数
        self.current_step = 0

        # 重置累积量
        self.cumulative_output_squared = 0
        self.cumulative_control_effort = 0

        # 获取初始状态
        self._update_state()

        return self.state.copy()

    def _update_state(self):
        """更新RL状态"""
        # 状态包括：当前输出Y，误差e，积分ei，微分ed，归一化时间
        normalized_time = self.current_step / self.episode_length

        self.state = np.array([
            self.state_space.Y[self.current_step] if self.current_step > 0 else 0,
            self.state_space.e,
            self.state_space.ei,
            self.state_space.ed,
            normalized_time
        ])

    def step(self, action):
        """
        执行一步动作

        Args:
            action: 归一化的PID参数 [kp, ki, kd] 在[-1, 1]范围内

        Returns:
            next_state, reward, done, info
        """
        if self.state_space is None:
            raise ValueError("StateSpace instance not set. Call set_state_space first.")

        # 将动作转换为实际PID参数
        pid_params = self.config.denormalize_action(action)
        kp, ki, kd = pid_params

        # 更新StateSpace的PID参数
        self.state_space.kp = kp
        self.state_space.ki = ki
        self.state_space.kd = kd

        # 执行StateSpace的一步计算
        if self.current_step < self.episode_length - 1:
            # 计算当前步的控制力和输出
            X_col = self.state_space.X[:, self.current_step].reshape(8, 1)

            # PID计算
            self.state_space.ei += self.state_space.e * self.dt
            self.state_space.ed = (self.state_space.e - self.state_space.e_last) / self.dt
            self.state_space.e_last = self.state_space.e
            u = kp * self.state_space.e + ki * self.state_space.ei + kd * self.state_space.ed

            # 输出计算
            Y = self.state_space.C @ X_col
            self.state_space.Y[self.current_step] = Y
            self.state_space.e = Y

            # 噪声计算
            F1 = self.state_space._StateSpace__compute_noise(self.current_step)
            F2 = self.state_space._StateSpace__compute_noise(self.current_step + 1)

            # 确保形状正确
            if F1.ndim == 1 or F1.shape != (8, 1):
                F1 = F1.reshape(8, 1)
            if F2.ndim == 1 or F2.shape != (8, 1):
                F2 = F2.reshape(8, 1)

            # 状态导数
            Xd_col = self.state_space.A @ X_col + self.state_space.B * u + F1
            self.state_space.Xd[:, self.current_step] = Xd_col.reshape(-1)

            # RK4积分
            k1_col = self.dt * Xd_col
            k2_col = self.dt * (self.state_space.A @ (X_col + k1_col / 2) +
                                self.state_space.B * u + (F1 + F2) / 2)
            k3_col = self.dt * (self.state_space.A @ (X_col + k2_col / 2) +
                                self.state_space.B * u + (F1 + F2) / 2)
            k4_col = self.dt * (self.state_space.A @ (X_col + k3_col) +
                                self.state_space.B * u + F2)

            X_update_col = X_col + (k1_col + 2 * k2_col + 2 * k3_col + k4_col) / 6.0
            self.state_space.X[:, self.current_step + 1] = X_update_col.reshape(-1)

            # 更新步数
            self.current_step += 1

            # 计算奖励
            reward = self._calculate_reward(Y, u, pid_params)

            # 检查是否结束
            done = self.current_step >= self.episode_length - 1

            # 更新状态
            self._update_state()

            info = {
                'output': float(Y),
                'control_force': float(u),
                'kp': kp,
                'ki': ki,
                'kd': kd
            }

            return self.state.copy(), reward, done, info
        else:
            # Episode结束
            done = True
            reward = 0
            info = {}
            return self.state.copy(), reward, done, info

    def _calculate_reward(self, output, control_force, pid_params):
        """计算奖励"""
        # 输出平方惩罚（希望输出趋近于0）
        output_penalty = self.config.OUTPUT_WEIGHT * (output ** 2)

        # 控制力惩罚（避免过大控制力）
        control_penalty = self.config.CONTROL_WEIGHT * (control_force ** 2)

        # PID参数变化惩罚（鼓励平滑的参数变化）
        param_change = np.sum((pid_params - self.last_pid_params) ** 2)
        param_penalty = self.config.PARAM_PENALTY * param_change

        # 更新上次参数
        self.last_pid_params = pid_params.copy()

        # 累积用于分析
        self.cumulative_output_squared += output ** 2
        self.cumulative_control_effort += control_force ** 2

        # 总奖励（负的成本）
        reward = - (output_penalty + control_penalty + param_penalty)

        return reward

    def get_performance_metrics(self):
        """获取性能指标"""
        return {
            'cumulative_output_squared': self.cumulative_output_squared,
            'cumulative_control_effort': self.cumulative_control_effort,
            'total_cost': self.cumulative_output_squared + self.config.CONTROL_WEIGHT * self.cumulative_control_effort
        }
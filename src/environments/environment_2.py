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

        # 初始化参数记录
        self.state_space.kp = np.ones(self.episode_length) * self.config.KP_RANGE[0]
        self.state_space.ki = np.ones(self.episode_length) * self.config.KI_RANGE[0]
        self.state_space.kd = np.ones(self.episode_length) * self.config.KD_RANGE[0]

    def reset(self, initial_state=None):
        """重置环境 - 必须与StateSpace的初始状态逻辑完全一致"""
        if self.state_space is None:
            raise ValueError("StateSpace instance not set.")

        # 1. 重置物理状态 X
        self.state_space.X = np.zeros((4, self.config.EPISODE_LENGTH))
        self.state_space.Xd = np.zeros((4, self.config.EPISODE_LENGTH))  # 确保Xd也被重置

        if initial_state is not None:
            self.state_space.X[:, 0] = initial_state

        # 2. 重置 PID 状态
        self.state_space.e = 0
        self.state_space.ei = 0
        self.state_space.ed = 0
        self.state_space.u = 0  # 初始控制力为0

        # 3. 初始时刻的预计算 (对应 solve 循环开始前的状态)
        # 在 solve 中，i=0 时，u 是 0 (self.u 的初始值)
        # 所以这里不需要额外操作，保持 u=0 即可

        self.current_step = 0
        self.cumulative_output_squared = 0
        self.cumulative_control_effort = 0

        # 生成初始观测
        self._update_rl_state()
        return self.state.copy()

    def _update_rl_state(self):
        """
        更新返回给RL的状态向量
        注意：RL通常根据当前观测 S_t 决定 A_t。
        这里我们返回刚刚计算完的状态。
        """
        # 使用 current_step (此时已经指向 i+1，或者初始为0)
        idx = self.current_step
        if idx >= self.episode_length:
            idx = self.episode_length - 1

        # 从 StateSpace 中读取数据
        # 注意：PID的内部状态 (e, ei, ed) 在 step 中是基于 idx-1 更新的
        # 为了给 RL 最新的信息，最好返回上一步计算的结果

        Y = float(self.state_space.Y[idx])  # 如果是step后，这里是 Y[i+1]

        # 关于 e, ei, ed:
        # 在 step 结束时，self.state_space.e 存储的是 e[i] (用于计算u[i]的那个误差)
        # 如果你想给 RL 看最新的状态，这里可能需要重新计算一下 e[i+1]
        # 但为了保持和 StateSpace 变量一致，我们直接读取

        e = float(self.state_space.e)
        ei = float(self.state_space.ei)
        ed = float(self.state_space.ed)

        normalized_time = float(idx / self.episode_length)

        self.state = np.array([Y, e, ei, ed, normalized_time])

    def _calculate_reward(self, output, control_force, pid_params):
        """
        计算奖励（实际上是负的代价函数）

        Args:
            output: 当前系统的位移/输出 Y (希望趋于0)
            control_force: 当前施加的控制力 u
            pid_params: 当前步的 [kp, ki, kd]
        """
        # 1. 输出惩罚：最核心的目标，让振动位移最小化
        # 使用平方项可以对大误差施加极高的惩罚，迫使系统快速收敛
        output_penalty = self.config.OUTPUT_WEIGHT * (output ** 2)

        # 2. 控制力惩罚：避免“控制力爆炸”和能源浪费
        # 在实际硬件中，作动器是有幅值限制的，过大的 u 会导致饱和甚至损坏
        control_penalty = self.config.CONTROL_WEIGHT * (control_force ** 2)

        # 3. PID参数变化惩罚（动作平滑性）
        # 这里建议对比当前 action 与上一次 last_pid_params 的差值
        # 你的原代码使用的是 ed**2，那其实是误差的变化速度，而不是参数的变化速度
        param_diff = np.sum((pid_params - self.last_pid_params) ** 2)
        param_penalty = self.config.PARAM_PENALTY * param_diff

        # 4. 生存奖励或额外惩罚（可选）
        # 如果系统发散（比如 output 超过了某个物理阈值），可以给予一个巨大的负奖励并提前结束
        stability_penalty = 0
        if abs(output) > 500.0:  # 假设物理极限
            stability_penalty = 1e6

        # 更新上次参数，供下一帧使用
        self.last_pid_params = pid_params.copy()

        # 累积用于分析
        self.cumulative_output_squared += output ** 2
        self.cumulative_control_effort += control_force ** 2

        # 总奖励 = 负的成本
        # RL 的目标是 Maximize Reward，即 Minimize Cost
        reward = - (output_penalty + control_penalty + param_penalty + stability_penalty)

        return float(reward)

    def step(self, action):
        """
        像素级复刻 StateSpace.solve 的执行顺序
        """
        if self.state_space is None:
            raise ValueError("StateSpace instance not set.")

        i = self.current_step
        if i >= self.episode_length - 1:
            return self.state.copy(), 0, True, {}

        # --- 0. 准备数据 ---
        # 获取当前 PID 参数
        pid_params = self.config.denormalize_action(action)
        kp, ki, kd = pid_params

        # 记录历史
        self.state_space.kp[i] = kp
        self.state_space.ki[i] = ki
        self.state_space.kd[i] = kd

        # 获取当前状态向量 (4, 1)
        X_col = self.state_space.X[:, i].reshape(4, 1)

        # 获取当前使用的 u (注意：这是上一步遗留的 u，即 u_{t-1})
        u_old = self.state_space.u

        # --- 1. 噪声计算 ---
        F1 = self.state_space.compute_noise(i)
        F2 = self.state_space.compute_noise(i + 1)
        # 形状修正
        if F1.ndim == 1: F1 = F1.reshape(4, 1)
        if F2.ndim == 1: F2 = F2.reshape(4, 1)

        # --- 2. 关键复刻：基于 旧u 计算 Xd 和 ed ---
        # 对应 StateSpace: Xd_col = self.A @ X_col + self.B * self.u + F1
        Xd_col = self.state_space.A @ X_col + self.state_space.B * u_old + F1
        self.state_space.Xd[:, i] = Xd_col.reshape(-1)

        # --- 3. PID 计算 ---
        # 计算 P 项
        self.state_space.e = (self.state_space.C @ X_col).item()

        # 计算 I 项 (注意：StateSpace 是先累加再用)
        self.state_space.ei += (self.state_space.e * self.dt)

        # 计算 D 项 (注意：StateSpace 用的是刚刚基于 旧u 算出来的 Xd)
        self.state_space.ed = (self.state_space.C @ Xd_col).item()

        # 计算 新u (u_t)
        # 符号完全照搬你的公式：正P，负I，负D
        u_new = kp * self.state_space.e - ki * self.state_space.ei - kd * self.state_space.ed

        # 更新类属性中的 u，供下一步RK4 (k2,k3,k4) 使用
        self.state_space.u = u_new

        # --- 4. 混合 RK4 积分 ---
        # 对应 StateSpace 的逻辑：
        # k1 使用 Xd_col (它包含 u_old)
        # k2, k3, k4 使用 self.state_space.u (它已经是 u_new)

        k1_col = self.dt * Xd_col

        # k2: 使用 u_new
        k2_col = self.dt * (self.state_space.A @ (X_col + k1_col / 2) +
                            self.state_space.B * u_new + (F1 + F2) / 2)

        # k3: 使用 u_new
        k3_col = self.dt * (self.state_space.A @ (X_col + k2_col / 2) +
                            self.state_space.B * u_new + (F1 + F2) / 2)

        # k4: 使用 u_new
        k4_col = self.dt * (self.state_space.A @ (X_col + k3_col) +
                            self.state_space.B * u_new + F2)

        # 状态更新
        X_update_col = X_col + (k1_col + 2 * k2_col + 2 * k3_col + k4_col) / 6.0
        self.state_space.X[:, i + 1] = X_update_col.reshape(-1)

        # --- 5. 更新输出 Y ---
        # StateSpace是在循环末尾计算 Y[i+1]
        Y_next = (self.state_space.C @ X_update_col).item()
        self.state_space.Y[i + 1] = Y_next

        # --- 6. 收尾工作 ---
        self.current_step += 1

        # 计算奖励 (使用当前的 Y 和 u_new)
        reward = self._calculate_reward(self.state_space.Y[i], u_new, pid_params)

        done = self.current_step >= self.episode_length - 1

        # 更新 RL 观测
        self._update_rl_state()

        info = {
            'output': float(self.state_space.Y[i]),
            'control_force': float(u_new),
            'kp': kp, 'ki': ki, 'kd': kd
        }

        return self.state.copy(), reward, done, info
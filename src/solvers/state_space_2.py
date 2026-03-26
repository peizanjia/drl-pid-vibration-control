import numpy as np
import matplotlib.pyplot as plt
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# X'(t) = A X(t) + B u(t) + F(t)
# Y(t) = C X(t)


class StateSpace:

    def __init__(self, config, noise_term, dt=0.01, tn=10000, dx=0.005, kp=0.0, ki=0.0, kd=0.0):
        self.config = config
        self.noise_term = noise_term
        self.dt = dt
        self.tn = tn
        self.dx = dx
        self.Reference = np.zeros(tn)
        self.current_step = 0

        # PID gains (initial)
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # PID internal states
        self.e = 0
        self.ei = 0
        self.ed = 0

        # State and output arrays
        self.X = np.zeros((4, tn))
        self.Xd = np.zeros((4, tn))
        self.Y = np.zeros(tn)
        self.u = 0

        # Disturbance terms
        self.F1 = self.noise_term['F1']
        self.F2 = self.noise_term['F2']

        # System matrices
        self.A, self.B, self.C = self.assemble_mat()

        # External controller callback (set before solve if needed)
        self.external_controller_callback = None

        # History of PID gains for analysis
        self.kp_history = np.zeros(tn)
        self.ki_history = np.zeros(tn)
        self.kd_history = np.zeros(tn)

        self.u_history = np.zeros(tn)

    def set_external_controller(self, controller_callback):
        """
        Register an external controller callback.

        Args:
            controller_callback: function(Y, time) -> (kp, ki, kd)
        """
        self.external_controller_callback = controller_callback

    def assemble_mat(self):
        w1 = self.config['w1']
        w2 = self.config['w2']
        z1 = self.config['z1']
        z2 = self.config['z2']
        B1 = self.config['B1']
        B2 = self.config['B2']
        C1 = self.config['C1']
        C2 = self.config['C2']

        A = np.array([[0,        0,        1,            0           ],
                      [0,        0,        0,            1           ],
                      [-w1 ** 2, 0,        -2 * z1 * w1, 0           ],
                      [0,        -w2 ** 2, 0,            -2 * z2 * w2]])

        B = np.array([[0], [0], [B1], [B2]])

        C = np.array([C1, C2, 0, 0])

        return A, B, C

    def compute_noise(self, idx):
        if idx < self.tn:
            F = np.array(
                [[0], [0], [self.F1[idx, 1]], [self.F2[idx, 1]]])
        else:
            F = np.zeros((4, 1))
        return F

    def solve(self):
        # Ensure correct B shape
        if self.B.ndim == 1 or self.B.shape != (4, 1):
            self.B = self.B.reshape(4, 1)

        # If no external controller, use fixed PID gains
        if self.external_controller_callback is None:
            print("Warning: no external controller set. Using fixed PID gains.")

        for i in range(self.tn - 1):
            self.current_step = i
            # Record PID gains
            self.kp_history[i] = self.kp
            self.ki_history[i] = self.ki
            self.kd_history[i] = self.kd

            # Convert X[:, i] to a (4, 1) column vector
            X_col = self.X[:, i].reshape(4, 1)

            # --- 1) Disturbance terms ---
            F1 = self.compute_noise(i)
            F2 = self.compute_noise(i + 1)

            # Ensure F1/F2 are (4,1)
            if F1.ndim == 1 or F1.shape != (4, 1):
                F1 = F1.reshape(4, 1)
            if F2.ndim == 1 or F2.shape != (4, 1):
                F2 = F2.reshape(4, 1)

            # --- 2) Xdot ---
            Xd_col = self.A @ X_col + self.B * self.u + F1
            self.Xd[:, i] = Xd_col.reshape(-1)

            # --- 3) PID update ---
            self.e = (self.C @ X_col)
            self.ei += (self.e * self.dt)
            self.ed = (self.C @ Xd_col)

            self.u = self.kp * self.e - self.ki * self.ei - self.kd * self.ed
            self.u_history[i] = (self.u).item()

            # --- 4) External controller hook ---
            if self.external_controller_callback is not None:
                current_time = i * self.dt
                current_output = self.Y[i]

                new_kp, new_ki, new_kd = self.external_controller_callback(current_output, current_time)

                # Update gains for next step
                self.kp = new_kp
                self.ki = new_ki
                self.kd = new_kd

            # --- 5) RK4 integration ---
            k1_col = self.dt * Xd_col
            k2_col = self.dt * (self.A @ (X_col + k1_col / 2) + self.B * self.u + (F1 + F2) / 2)
            k3_col = self.dt * (self.A @ (X_col + k2_col / 2) + self.B * self.u + (F1 + F2) / 2)
            k4_col = self.dt * (self.A @ (X_col + k3_col) + self.B * self.u + F2)

            X_update_col = X_col + (k1_col + 2 * k2_col + 2 * k3_col + k4_col) / 6.0
            self.X[:, i + 1] = X_update_col.reshape(-1)

            # --- 6) Output ---
            Y_next = self.C @ X_update_col
            self.Y[i + 1] = Y_next.item()

        # Store final PID gains
        self.kp_history[-1] = self.kp
        self.ki_history[-1] = self.ki
        self.kd_history[-1] = self.kd

        self.u_history[-1] = self.u_history[-2]

    def plot_time_domain_response(self):
        """
        Plot tip displacement and sensor output over time.
        """
        # 1) Time axis (ensure dt is defined)
        N = self.X.shape[1]
        time_vector = np.arange(N) * self.dt

        # 2) Composite displacement
        z = -2 * self.X[0, :] + 2 * self.X[1, :]

        # 3) Plot layout
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        # Subplot 1: dynamics state
        ax1.plot(time_vector, z, color='#1f77b4', linewidth=1.5, label=r'$z = -2X_0 + 2X_1$')
        ax1.set_ylabel('Displacement (m)', fontsize=12)
        ax1.set_title('Tip Dynamics Response', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle=':', alpha=0.7)

        # Subplot 2: sensor output
        ax2.plot(time_vector, self.Y.flatten(), color='#d62728', linewidth=1.5, label='Sensor Output')
        ax2.set_ylabel('Voltage (V)', fontsize=12)
        ax2.set_title('Measured Sensor Signal', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle=':', alpha=0.7)

        # Subplot 3: control input
        ax3.plot(time_vector, self.u_history.flatten(), color='#2ca02c', linewidth=1.5, label='Control Input $u(t)$')
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.set_ylabel('u', fontsize=12)
        ax3.set_title('Control Input History', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, linestyle=':', alpha=0.7)

        # 4) Layout tweaks
        plt.tight_layout()

        # Save high-quality figure
        plt.savefig('../../results/state_response.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    from config.config import Config
    from config.config_loader import load_mat
    # 1) Base config
    config = Config()
    tn = config.EPISODE_LENGTH

    # 2) Load thermal data (optional)
    try:
        mt = load_mat()
    except Exception as e:
        print(f"Warning: mt_data load failed: {e}")
        mt = None

    # 3) Precompute projector coefficients
    from src.utils.utils import BeamDisturbanceProjector, create_noise_data

    proj = BeamDisturbanceProjector()
    projector_coeffs = proj.get_static_coeffs()

    # 4) Generate disturbance data (projector_data required)
    # Change option to 'impact' to check early impact behavior
    noise_data = create_noise_data(
        tn,
        option='maneuver',
        system_config=config.SYSTEM_CONFIG,
        mt_data=mt,
        projector_data=projector_coeffs
    )

    # 5) Instantiate state-space solver
    # Note: dx is spatial step; keep consistent with beam discretization
    state_space = StateSpace(
        config.SYSTEM_CONFIG,
        noise_data,
        dt=config.DT,
        tn=tn,
        kp=100,
        ki=30,
        kd=27
    )

    # 6) Solve and plot
    print("Solving state space equations...")
    state_space.solve()

    print("Plotting results...")
    state_space.plot_time_domain_response()

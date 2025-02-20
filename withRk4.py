import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Arc

# Parameters
L = 2.0  # Length of pendulum (m)
g = 9.81  # Gravitational acceleration (m/s^2)
dt = 0.01  # Time step (s)
T = 50.0  # Total time (s)
theta0 = np.pi/3  # Initial angle (radians)
omega0 = 0.0  # Initial angular velocity (rad/s)

# Time array
t = np.arange(0, T, dt)

# RK4 method function
def rk4_step(theta, omega, dt):
    # Define the system of first-order ODEs
    def f(state):
        theta, omega = state
        return np.array([omega, -(g/L) * np.sin(theta)])
    
    state = np.array([theta, omega])
    
    k1 = f(state)
    k2 = f(state + dt*k1/2)
    k3 = f(state + dt*k2/2)
    k4 = f(state + dt*k3)
    
    state_new = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    return state_new[0], state_new[1]

# Calculate trajectories for both methods
theta_euler = np.zeros_like(t)
omega_euler = np.zeros_like(t)
theta_rk4 = np.zeros_like(t)
omega_rk4 = np.zeros_like(t)

# Initial conditions
theta_euler[0] = theta_rk4[0] = theta0
omega_euler[0] = omega_rk4[0] = omega0

# Calculate both solutions
for i in range(1, len(t)):
    # Euler method
    omega_euler[i] = omega_euler[i-1] - (g/L) * np.sin(theta_euler[i-1]) * dt
    theta_euler[i] = theta_euler[i-1] + omega_euler[i] * dt
    
    # RK4 method
    theta_rk4[i], omega_rk4[i] = rk4_step(theta_rk4[i-1], omega_rk4[i-1], dt)

# Set up the figure and animation
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])

# Pendulum animation plot (spanning both columns)
ax_anim = fig.add_subplot(gs[0, :])
ax_euler = fig.add_subplot(gs[1, 0])
ax_rk4 = fig.add_subplot(gs[1, 1])

# Set up pendulum animation plot
ax_anim.set_xlim(-2.5, 2.5)
ax_anim.set_ylim(-2.5, 2.5)
ax_anim.set_aspect('equal')
ax_anim.grid(True, alpha=0.3)
ax_anim.set_title('Pendulum Animation')

# Add reference lines for pendulum
ax_anim.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax_anim.axvline(x=0, color='gray', linestyle='--', alpha=0.5)


# Set up Euler method graph
ax_euler.set_xlim(0, T)
ax_euler.set_ylim(-np.pi, np.pi)
ax_euler.set_xlabel('Time (s)')
ax_euler.set_ylabel('Angle (radians)')
ax_euler.grid(True, alpha=0.3)
ax_euler.set_title("Euler's Method")

# Set up RK4 method graph
ax_rk4.set_xlim(0, T)
ax_rk4.set_ylim(-np.pi, np.pi)
ax_rk4.set_xlabel('Time (s)')
ax_rk4.set_ylabel('Angle (radians)')
ax_rk4.grid(True, alpha=0.3)
ax_rk4.set_title('RK4 Method')

# Add reference lines for both graphs
for ax in [ax_euler, ax_rk4]:
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Equilibrium')
    ax.axhline(y=theta0, color='r', linestyle='--', alpha=0.3, label='Initial angle')
    ax.axhline(y=-theta0, color='r', linestyle='--', alpha=0.3)
    ax.legend()

# Initialize pendulum objects
pivot = Circle((0, 0), 0.1, color='black')
ax_anim.add_patch(pivot)

# Pendulum line and bob
line, = ax_anim.plot([], [], 'k-', lw=2)
bob = Circle((0, 0), 0.2, color='red')
ax_anim.add_patch(bob)

# Initialize graph objects
euler_line, = ax_euler.plot([], [], 'b-')
euler_point, = ax_euler.plot([], [], 'ro')
rk4_line, = ax_rk4.plot([], [], 'g-')
rk4_point, = ax_rk4.plot([], [], 'ro')

def init():
    line.set_data([], [])
    bob.center = (0, 0)
    euler_line.set_data([], [])
    euler_point.set_data([], [])
    rk4_line.set_data([], [])
    rk4_point.set_data([], [])
    return line, bob, euler_line, euler_point, rk4_line, rk4_point

def animate(frame):
    # Update pendulum (using RK4 solution for animation)
    x = L * np.sin(theta_rk4[frame])
    y = -L * np.cos(theta_rk4[frame])
    
    line.set_data([0, x], [0, y])
    bob.center = (x, y)
    
    # Update both graphs
    euler_line.set_data(t[:frame], theta_euler[:frame])
    euler_point.set_data([t[frame]], [theta_euler[frame]])
    
    rk4_line.set_data(t[:frame], theta_rk4[:frame])
    rk4_point.set_data([t[frame]], [theta_rk4[frame]])
    
    return line, bob, euler_line, euler_point, rk4_line, rk4_point

# Create animation
anim = FuncAnimation(fig, animate, init_func=init, frames=len(t),
                    interval=dt*1000, blit=True)

plt.tight_layout()
plt.show()
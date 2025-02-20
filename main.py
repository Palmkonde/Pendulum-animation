import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Arc

# Parameters
L = 2.0  # Length of pendulum (m)
g = 9.81  # Gravitational acceleration (m/s^2)
dt = 0.02  # Time step (s)
T = 10.0  # Total time (s)
theta0 = np.pi/2  # Initial angle (radians)
omega0 = 0.0  # Initial angular velocity (rad/s)

# Calculate full trajectory first
t = np.arange(0, T, dt)
theta = np.zeros_like(t)
omega = np.zeros_like(t)
theta[0] = theta0
omega[0] = omega0

# Euler's method
for i in range(1, len(t)):
    omega[i] = omega[i-1] - (g/L) * np.sin(theta[i-1]) * dt
    theta[i] = theta[i-1] + omega[i] * dt

# Set up the figure and animation
fig = plt.figure(figsize=(15, 6))
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])  # Pendulum animation
ax2 = fig.add_subplot(gs[0, 1])  # Graph

# Set up pendulum plot
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-2.5, 2.5)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# Add horizontal and vertical reference lines for pendulum
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# Set up graph
ax2.set_xlim(0, T)
ax2.set_ylim(-np.pi, np.pi)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angle (radians)')
ax2.grid(True, alpha=0.3)

# Add reference lines for important angles
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Equilibrium')
ax2.axhline(y=theta0, color='r', linestyle='--', alpha=0.3, label='Initial angle')
ax2.axhline(y=-theta0, color='r', linestyle='--', alpha=0.3)
ax2.legend()

# Initialize pendulum objects
pivot = Circle((0, 0), 0.1, color='black')
ax1.add_patch(pivot)

# Pendulum line and bob
line, = ax1.plot([], [], 'k-', lw=2)
bob = Circle((0, 0), 0.2, color='red')
ax1.add_patch(bob)

# Initialize graph objects
graph_line, = ax2.plot([], [], 'b-')
point, = ax2.plot([], [], 'ro')

# Animation initialization function
def init():
    line.set_data([], [])
    bob.center = (0, 0)
    graph_line.set_data([], [])
    point.set_data([], [])
    return line, bob, graph_line, point

# Animation update function
def animate(frame):
    # Update pendulum
    x = L * np.sin(theta[frame])
    y = -L * np.cos(theta[frame])
    
    line.set_data([0, x], [0, y])
    bob.center = (x, y)
    
    # Update graph
    graph_line.set_data(t[:frame], theta[:frame])
    point.set_data([t[frame]], [theta[frame]])
    
    return line, bob, graph_line, point

# Create animation
anim = FuncAnimation(fig, animate, init_func=init, frames=len(t),
                    interval=dt*1000, blit=True)

plt.tight_layout()
plt.show()
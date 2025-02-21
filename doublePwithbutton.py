import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# Constants
G = 9.81  # gravitational acceleration
L1 = 1.0  # length of pendulum 1
L2 = 1.0  # length of pendulum 2
M1 = 1.0  # mass of pendulum 1
M2 = 1.0  # mass of pendulum 2
dt = 0.01  # time step
t_end = 20  # end time

# Colors
COLOR_BOB1 = 'blue'
COLOR_BOB2 = 'red'
COLOR_ROD = 'black'

def derivatives(state):
    theta1, omega1, theta2, omega2 = state
    
    # Helper calculations
    delta = theta2 - theta1
    den = (M1 + M2) * L1 - M2 * L1 * np.cos(delta) * np.cos(delta)
    
    # Differential equations
    dtheta1 = omega1
    domega1 = (M2 * L1 * omega1 * omega1 * np.sin(delta) * np.cos(delta) 
               + M2 * G * np.sin(theta2) * np.cos(delta) 
               + M2 * L2 * omega2 * omega2 * np.sin(delta) 
               - (M1 + M2) * G * np.sin(theta1)) / den
    
    dtheta2 = omega2
    domega2 = (-M2 * L2 * omega2 * omega2 * np.sin(delta) * np.cos(delta) 
               + (M1 + M2) * (G * np.sin(theta1) * np.cos(delta) 
               - L1 * omega1 * omega1 * np.sin(delta) 
               - G * np.sin(theta2))) / (den)
    
    return np.array([dtheta1, domega1, dtheta2, domega2])

def euler_step(state):
    return state + dt * derivatives(state)

def rk4_step(state):
    k1 = derivatives(state)
    k2 = derivatives(state + dt * k1/2)
    k3 = derivatives(state + dt * k2/2)
    k4 = derivatives(state + dt * k3)
    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

# Time array
t = np.arange(0, t_end, dt)
n_steps = len(t)

# Initial conditions [theta1, omega1, theta2, omega2]
initial_state = np.array([np.pi/2, 0, np.pi/2, 0])

# Arrays to store solutions
euler_sol = np.zeros((n_steps, 4))
rk4_sol = np.zeros((n_steps, 4))
euler_sol[0] = initial_state
rk4_sol[0] = initial_state

# Solve using both methods
for i in range(1, n_steps):
    euler_sol[i] = euler_step(euler_sol[i-1])
    rk4_sol[i] = rk4_step(rk4_sol[i-1])

# Calculate positions for both methods
x1_euler = L1 * np.sin(euler_sol[:, 0])
y1_euler = -L1 * np.cos(euler_sol[:, 0])
x2_euler = x1_euler + L2 * np.sin(euler_sol[:, 2])
y2_euler = y1_euler - L2 * np.cos(euler_sol[:, 2])

x1_rk4 = L1 * np.sin(rk4_sol[:, 0])
y1_rk4 = -L1 * np.cos(rk4_sol[:, 0])
x2_rk4 = x1_rk4 + L2 * np.sin(rk4_sol[:, 2])
y2_rk4 = y1_rk4 - L2 * np.cos(rk4_sol[:, 2])

# Create figure and subplots with space for button
fig = plt.figure(figsize=(15, 13))  # Made slightly taller for button
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.1])  # Added space for button

# Animation subplots
ax_anim_euler = fig.add_subplot(gs[0, 0])
ax_anim_rk4 = fig.add_subplot(gs[0, 1])
for ax in [ax_anim_euler, ax_anim_rk4]:
    ax.set_xlim(-(L1 + L2), (L1 + L2))
    ax.set_ylim(-(L1 + L2), (L1 + L2))
    ax.set_aspect('equal')
    ax.grid(True)

ax_anim_euler.set_title('Double Pendulum (Euler)')
ax_anim_rk4.set_title('Double Pendulum (RK4)')

# Graph subplots
ax_graph_euler = fig.add_subplot(gs[1, 0])
ax_graph_rk4 = fig.add_subplot(gs[1, 1])
for ax in [ax_graph_euler, ax_graph_rk4]:
    ax.set_xlim(0, t_end)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (rad)')
    ax.grid(True)

ax_graph_euler.set_title('Angles Evolution (Euler)')
ax_graph_rk4.set_title('Angles Evolution (RK4)')

# Button axis
ax_button = plt.axes([0.4, 0.02, 0.2, 0.04])  # [left, bottom, width, height]
button = Button(ax_button, 'Play/Pause')

# Initialize animation elements for Euler
line1_euler, = ax_anim_euler.plot([], [], COLOR_ROD, lw=2)
bob1_euler, = ax_anim_euler.plot([], [], 'o', color=COLOR_BOB1, markersize=10, label='Bob 1')
line2_euler, = ax_anim_euler.plot([], [], COLOR_ROD, lw=2)
bob2_euler, = ax_anim_euler.plot([], [], 'o', color=COLOR_BOB2, markersize=10, label='Bob 2')
trace_euler, = ax_anim_euler.plot([], [], 'r-', alpha=0.3)

# Initialize animation elements for RK4
line1_rk4, = ax_anim_rk4.plot([], [], COLOR_ROD, lw=2)
bob1_rk4, = ax_anim_rk4.plot([], [], 'o', color=COLOR_BOB1, markersize=10, label='Bob 1')
line2_rk4, = ax_anim_rk4.plot([], [], COLOR_ROD, lw=2)
bob2_rk4, = ax_anim_rk4.plot([], [], 'o', color=COLOR_BOB2, markersize=10, label='Bob 2')
trace_rk4, = ax_anim_rk4.plot([], [], 'r-', alpha=0.3)

# Initialize angle plots
line_euler_theta1, = ax_graph_euler.plot([], [], label='θ1', color=COLOR_BOB1)
line_euler_theta2, = ax_graph_euler.plot([], [], label='θ2', color=COLOR_BOB2)
line_rk4_theta1, = ax_graph_rk4.plot([], [], label='θ1', color=COLOR_BOB1)
line_rk4_theta2, = ax_graph_rk4.plot([], [], label='θ2', color=COLOR_BOB2)

# Add legends
ax_graph_euler.legend()
ax_graph_rk4.legend()

# Time text
time_template = 'time = %.1fs'
time_text_euler = ax_anim_euler.text(0.05, 0.95, '', transform=ax_anim_euler.transAxes)
time_text_rk4 = ax_anim_rk4.text(0.05, 0.95, '', transform=ax_anim_rk4.transAxes)

# Animation traces
trace_x_euler, trace_y_euler = [], []
trace_x_rk4, trace_y_rk4 = [], []
plot_t = []
plot_euler_theta1, plot_euler_theta2 = [], []
plot_rk4_theta1, plot_rk4_theta2 = [], []

# Animation control variables
paused = True
first_click = True

def handle_click(event):
    global paused, first_click
    if first_click:
        anim.event_source.start()
        first_click = False
    paused = not paused
    if paused:
        anim.event_source.stop()
    else:
        anim.event_source.start()

# Connect the button to the handler
button.on_clicked(handle_click)

def animate(i):
    if paused:
        return (line1_euler, bob1_euler, line2_euler, bob2_euler, trace_euler,
                line1_rk4, bob1_rk4, line2_rk4, bob2_rk4, trace_rk4,
                time_text_euler, time_text_rk4,
                line_euler_theta1, line_euler_theta2,
                line_rk4_theta1, line_rk4_theta2)

    # Update Euler pendulum
    line1_euler.set_data([0, x1_euler[i]], [0, y1_euler[i]])
    bob1_euler.set_data([x1_euler[i]], [y1_euler[i]])
    line2_euler.set_data([x1_euler[i], x2_euler[i]], [y1_euler[i], y2_euler[i]])
    bob2_euler.set_data([x2_euler[i]], [y2_euler[i]])
    
    # Update RK4 pendulum
    line1_rk4.set_data([0, x1_rk4[i]], [0, y1_rk4[i]])
    bob1_rk4.set_data([x1_rk4[i]], [y1_rk4[i]])
    line2_rk4.set_data([x1_rk4[i], x2_rk4[i]], [y1_rk4[i], y2_rk4[i]])
    bob2_rk4.set_data([x2_rk4[i]], [y2_rk4[i]])
    
    # Update traces
    trace_x_euler.append(x2_euler[i])
    trace_y_euler.append(y2_euler[i])
    trace_euler.set_data(trace_x_euler, trace_y_euler)
    
    trace_x_rk4.append(x2_rk4[i])
    trace_y_rk4.append(y2_rk4[i])
    trace_rk4.set_data(trace_x_rk4, trace_y_rk4)
    
    # Update time
    plot_t.append(t[i])
    
    # Update angle plots
    plot_euler_theta1.append(euler_sol[i, 0])
    plot_euler_theta2.append(euler_sol[i, 2])
    line_euler_theta1.set_data(plot_t, plot_euler_theta1)
    line_euler_theta2.set_data(plot_t, plot_euler_theta2)
    
    plot_rk4_theta1.append(rk4_sol[i, 0])
    plot_rk4_theta2.append(rk4_sol[i, 2])
    line_rk4_theta1.set_data(plot_t, plot_rk4_theta1)
    line_rk4_theta2.set_data(plot_t, plot_rk4_theta2)
    
    # Update time text
    time_text_euler.set_text(time_template % (t[i]))
    time_text_rk4.set_text(time_template % (t[i]))
    
    return (line1_euler, bob1_euler, line2_euler, bob2_euler, trace_euler,
            line1_rk4, bob1_rk4, line2_rk4, bob2_rk4, trace_rk4,
            time_text_euler, time_text_rk4,
            line_euler_theta1, line_euler_theta2,
            line_rk4_theta1, line_rk4_theta2)

# Create animation that starts paused
anim = FuncAnimation(fig, animate, frames=n_steps, 
                    interval=20, blit=True)
anim.event_source.stop()  # Start paused

plt.tight_layout()
plt.show()
import pybullet as p
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Lévy‐walk parameters
alpha, scale = 1.5, 0.1
num_steps, dt = 1000, 1/240.0

# Physics parameters for smoothing and drag
inertia_factor = 0.9       # how much of previous velocity is retained (0–1)
drag_coefficient = 0.05    # simple linear drag: F_drag = –k * v
max_force = 1e-4           # cap random force magnitude

slowdown_factor = 5  # slow down the animation

# --- PyBullet setup ---
p.connect(p.DIRECT)
p.setGravity(0, 0, -9.81)

radius, mass = 0.01, 0.001
col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1,0,0,1])
body = p.createMultiBody(baseMass=mass,
                         baseCollisionShapeIndex=col,
                         baseVisualShapeIndex=vis,
                         basePosition=[0.5, 0.5, 0.5])

# initialize velocity
current_v = np.zeros(3)
positions = []

for step in range(num_steps):
    # 1) Sample a Lévy‐walk “desired step” as a force direction
    theta = np.random.rand() * 2 * np.pi
    phi   = np.arccos(2 * np.random.rand() - 1)
    step_size = (np.random.pareto(alpha) + 1) * scale
    # convert to a force vector
    random_dir = np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    F_random = np.clip(random_dir * step_size, -max_force, max_force)

    # 2) Compute drag force
    F_drag = -drag_coefficient * current_v

    # 3) Net force and acceleration
    F_net = F_random + F_drag + np.array([0,0, mass * 9.81])  # counteract gravity so bug stays aloft
    accel = F_net / mass

    # 4) Update velocity with inertia smoothing
    current_v = inertia_factor * current_v + (1 - inertia_factor) * (current_v + accel * dt)

    # 5) Apply external force on the body each step
    p.applyExternalForce(body,
                         linkIndex=-1,
                         forceObj=F_net.tolist(),
                         posObj=[0,0,0],
                         flags=p.LINK_FRAME)
    p.stepSimulation()

    # 6) Record position
    pos, _ = p.getBasePositionAndOrientation(body)
    positions.append(pos)

p.disconnect()

# Unpack into coordinate lists
xs, ys, zs = zip(*positions)

# --- Static 3D Plot ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs, ys, zs, label='Trajectory')
ax.scatter(xs, ys, zs, c='r', s=2)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend()
plt.show()

# --- Animated 3D Trajectory ---
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
line, = ax2.plot([], [], [], lw=2)

ax2.set_xlim(min(xs), max(xs))
ax2.set_ylim(min(ys), max(ys))
ax2.set_zlim(min(zs), max(zs))

def update_frame(i):
    line.set_data(xs[:i], ys[:i])
    line.set_3d_properties(zs[:i])
    return line,

interval_ms = dt * 1000 * slowdown_factor
ani = animation.FuncAnimation(
    fig2,
    update_frame,
    frames=len(xs),
    interval=interval_ms,
    blit=True
)

plt.show()

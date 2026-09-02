import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import json
import sys

SIM = "mass_spectrometer"
if len(sys.argv) > 1:
    SIM = sys.argv[1]


spawn_timer = 0
KILL_MARGIN = 5
SIM_TIME = 0
pressure_impulse = 0.0
PRESSURE_WINDOW = 1.0  
pressure_timer = PRESSURE_WINDOW + 1


with open(f"/home/shepad/physics/physics_sim/configs_uniform/{SIM}.json", "r") as f:
    cfg = json.load(f)

BIGS = cfg["BIGS"]
STEP = cfg["STEP"]

OPPOSITE_CHARGE = cfg["OPPOSITE_CHARGE"] == 1

BIG_MASS = cfg["BIG_MASS"]
SMALL_MASS = cfg["SMALL_MASS"]

AMOUNT_IN_X = cfg["AMOUNT_IN_X"]
AMOUNT_IN_Y = cfg["AMOUNT_IN_Y"]

TEMP = cfg["TEMP"]

FIELD_STRENGTH = cfg["FIELD_STRENGTH"]
E_FIELD = np.array(cfg["E_FIELD"], dtype=float)

SPAWN_RATE = cfg["SPAWN_RATE"]

MAX_PATH_POINTS = cfg["MAX_PATH_POINTS"]
KILL_MARGIN = cfg["KILL_MARGIN"]

CHARGE = cfg["CHARGE"]

WALL_COLISIONS = cfg["WALL_COLISIONS"] == 1
PARTICLE_COLISSIONS = cfg["PARTICLE_COLISIONS"] == 1

NEW_BALL_SPAWNS = cfg["NEW_BALL_SPAWNS"] == 1
AVG_SPAWN_VELO = cfg["AVG_SPAWN_VELO"]


def kinetic_energy(particles):
    return sum(0.5 * p.mass * np.dot(p.velocity, p.velocity)
               for p in particles)

def total_momentum(particles):
    P = np.zeros(2)
    for p in particles:
        P += p.mass * p.velocity
    return P


def initialize_new_ball(system):
    """
    Create a new particle off-screen, heading into the box.
    Mass is randomly chosen according to the big/small ratio.
    """
    # spawn off the left edge
    x = system.xmin - 1.0
    y = np.random.uniform(system.ymin, system.ymax)

    # mass according to big/small ratio
    big_ratio = BIGS / (AMOUNT_IN_X * AMOUNT_IN_Y)
    mass = BIG_MASS if np.random.rand() < big_ratio else SMALL_MASS

    # velocity pointing roughly to the right
    speed_mean = AVG_SPAWN_VELO 
    speed_std = 1.0
    vx = np.random.normal(loc=speed_mean, scale=speed_std)
    vy = np.random.normal(loc=0.0, scale=3.0)

    radius = 0.12 * (mass ** (1/3))
    new_particle = Particle(mass, [x, y], [vx, vy], radius)

    system.particles.append(new_particle)
    return new_particle


def rotate_velocity(v, omega_dt):
    """Exact rotation of velocity in a uniform magnetic field"""
    c = np.cos(omega_dt)
    s = np.sin(omega_dt)
    return np.array([
        c*v[0] - s*v[1],
        s*v[0] + c*v[1]
    ])

def discrete_magnetic(v, dt):
    lst = np.array(list(v) + [0])
    lst = np.cross(lst, np.array([0,0,1]))
    return v + np.array(lst[0], lst[1]) * dt

# 
class Particle:
    def __init__(self, mass, position, velocity, radius):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.radius = radius

# 
class ElasticMagneticBoxSystem:
    def __init__(self, Nx=AMOUNT_IN_X, Ny=AMOUNT_IN_Y,
                 box=(0, 0, 10, 7.5), spacing=0.5):

        xs = np.linspace(box[0] + spacing, box[2] - spacing, Nx)
        ys = np.linspace(box[1] + spacing, box[3] - spacing, Ny)
        grid = [(x, y) for y in ys for x in xs]

        self.xmin, self.ymin, self.xmax, self.ymax = box
        self.N = len(grid)

        masses = np.array(
            ([BIG_MASS] * BIGS) +
            ([SMALL_MASS] * (self.N - BIGS))
        )

        radii = 0.12 * (masses ** (1/3))

        velocities = np.zeros((self.N, 2))
        for i, (x, y) in enumerate(grid):
            velocities[i] = [
                TEMP * np.sin(0.4*x + 0.2*i),
                TEMP * np.cos(0.5*y - 0.3*i)
            ]

        self.particles = [
            Particle(masses[i], grid[i], velocities[i], radii[i])
            for i in range(self.N)
        ]

    
    def step(self, dt):
        # magnetic
        for p in self.particles:
            omega = CHARGE * FIELD_STRENGTH / p.mass
            if p.mass == BIG_MASS and OPPOSITE_CHARGE:
                omega *= -1
            p.velocity = rotate_velocity(p.velocity, omega * dt) # boris transform for continous time
            # electric
            #p.velocity = discrete_magnetic(p.velocity, dt)
            if p.mass == BIG_MASS:
                p.velocity += CHARGE * E_FIELD * dt / p.mass
            else:
                p.velocity -= CHARGE * E_FIELD * dt / p.mass

        # move
        for p in self.particles:
            p.position += p.velocity * dt
        global pressure_impulse
        if WALL_COLISIONS:
            for p in self.particles:
                if p.position[0] - p.radius < self.xmin:
                    dp = 2 * p.mass * abs(p.velocity[0])
                    pressure_impulse += dp
                    p.position[0] = self.xmin + p.radius
                    p.velocity[0] *= -1

                if p.position[0] + p.radius > self.xmax:
                    dp = 2 * p.mass * abs(p.velocity[0])
                    pressure_impulse += dp
                    p.position[0] = self.xmax - p.radius
                    p.velocity[0] *= -1

                if p.position[1] - p.radius < self.ymin:
                    dp = 2 * p.mass * abs(p.velocity[0])
                    pressure_impulse += dp
                    p.position[1] = self.ymin + p.radius
                    p.velocity[1] *= -1

                if p.position[1] + p.radius > self.ymax:
                    dp = 2 * p.mass * abs(p.velocity[0])
                    pressure_impulse += dp
                    p.position[1] = self.ymax - p.radius
                    p.velocity[1] *= -1

        if PARTICLE_COLISSIONS:
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    A = self.particles[i]
                    B = self.particles[j]

                    delta = A.position - B.position
                    dist = np.linalg.norm(delta)
                    min_dist = A.radius + B.radius

                    if dist < min_dist:
                        n = delta / dist
                        v_rel = np.dot(A.velocity - B.velocity, n)
                        if v_rel > 0:
                            continue
                        J = (2 * v_rel) / (1/A.mass + 1/B.mass)
                        A.velocity -= (J / A.mass) * n
                        B.velocity += (J / B.mass) * n

                        overlap = min_dist - dist
                        A.position += n * (overlap / 2)
                        B.position -= n * (overlap / 2)

# animation
system = ElasticMagneticBoxSystem()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(system.xmin, system.xmax)
ax.set_ylim(system.ymin, system.ymax)
ax.set_aspect('equal')
ax.set_title("Elastic Brownian Motion with Magnetic Field and Tracers")
energy_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes,
    va="top", ha="left", fontsize=10
)

patches = []
paths = []       
histories = []   

for p in system.particles:
    color = "red" if p.mass == BIG_MASS else "blue"
    circle = plt.Circle(p.position, p.radius, color=color)
    patches.append(circle)
    ax.add_patch(circle)


    line, = ax.plot([], [], lw=1, alpha=0.5, color=color)
    paths.append(line)
    #histories.append([p.position.copy()])  # start with initial position
    histories.append(deque(maxlen=MAX_PATH_POINTS))

def update(frame):

    # remove particles far away for performance
    to_remove = []
    for i, p in enumerate(system.particles):
        if (p.position[0] < system.xmin - KILL_MARGIN or
            p.position[0] > system.xmax + KILL_MARGIN or
            p.position[1] < system.ymin - KILL_MARGIN or
            p.position[1] > system.ymax + KILL_MARGIN):
            to_remove.append(i)

    # remove in reverse order to keep indices correct
    for i in reversed(to_remove):
        system.particles.pop(i)
        patches.pop(i)
        paths.pop(i)
        histories.pop(i)

    global SIM_TIME
    global spawn_timer
    spawn_timer += STEP
    SIM_TIME += STEP

    global pressure_impulse, pressure_timer, last_pressure

    pressure_timer += STEP

    if pressure_timer >= PRESSURE_WINDOW:
        wall_length = 2 * ((system.xmax - system.xmin) +
                       (system.ymax - system.ymin))

        pressure = pressure_impulse / (pressure_timer * wall_length)

        pressure_impulse = 0.0
        pressure_timer = 0.0
        last_pressure = pressure


    if spawn_timer >= SPAWN_RATE and NEW_BALL_SPAWNS:
        new_particle = initialize_new_ball(system)
    
        # add circle patch for visualization
        color = "red" if new_particle.mass == BIG_MASS else "blue"
        circle = plt.Circle(new_particle.position, new_particle.radius, color=color)
        patches.append(circle)
        ax.add_patch(circle)

        # add tracer line for the new particle
        line, = ax.plot([], [], lw=1, alpha=0.5, color=color)
        paths.append(line)
        #histories.append([new_particle.position.copy()])
        histories.append(deque(maxlen=MAX_PATH_POINTS))


        spawn_timer = 0.0  # reset timer


    system.step(STEP)
    KE = kinetic_energy(system.particles)
    P = total_momentum(system.particles)
    energy_text.set_text(
    f"KE = {KE:.2f}\n"
    f"P = ({P[0]:.2f}, {P[1]:.2f})\n"
    f"presure = {last_pressure:.3f}"
    )


    for p, circle, history, path in zip(system.particles, patches, histories, paths):
        circle.center = p.position

        
        history.append(p.position.copy())
        xs, ys = zip(*history)
        path.set_data(xs, ys)

    return patches + paths + [energy_text]

anim = FuncAnimation(
    fig, update, frames=2000, interval=4, blit=True
)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

spawn_timer = 0
KILL_MARGIN = 5

# to mess with
BIGS = 1
STEP = 0.01

BIG_MASS = 5.0
SMALL_MASS = 1.0

AMOUNT_IN_X = 2
AMOUNT_IN_Y = 1

TEMP = 1.5
CHARGE = 1.0   #abs val    

E_FIELD = np.array([0, 2])
FIELD_STRENGTH = 0 # magnetic

WALL_COLISIONS = True
PARTICLE_COLISSIONS = True

NEW_BALL_SPAWNS = False
SPAWN_RATE = 1
AVG_SPAWN_VELO = 3

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
    vy = np.random.normal(loc=0.0, scale=1.0)

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
            if p.mass == BIG_MASS:
                omega *= -1
            p.velocity = rotate_velocity(p.velocity, omega * dt)
            # electric
            if p.mass == BIG_MASS:
                p.velocity += CHARGE * E_FIELD * dt / p.mass
            else:
                p.velocity -= CHARGE * E_FIELD * dt / p.mass

        # move
        for p in self.particles:
            p.position += p.velocity * dt

        if WALL_COLISIONS:
            for p in self.particles:
                if p.position[0] - p.radius < self.xmin:
                    p.position[0] = self.xmin + p.radius
                    p.velocity[0] *= -1

                if p.position[0] + p.radius > self.xmax:
                    p.position[0] = self.xmax - p.radius
                    p.velocity[0] *= -1

                if p.position[1] - p.radius < self.ymin:
                    p.position[1] = self.ymin + p.radius
                    p.velocity[1] *= -1

                if p.position[1] + p.radius > self.ymax:
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
    histories.append([p.position.copy()])  # start with initial position

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


    global spawn_timer
    spawn_timer += STEP

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
        histories.append([new_particle.position.copy()])

        spawn_timer = 0.0  # reset timer


    system.step(STEP)
    for p, circle, history, path in zip(system.particles, patches, histories, paths):
        circle.center = p.position

        
        history.append(p.position.copy())
        xs, ys = zip(*history)
        path.set_data(xs, ys)

    return patches + paths   

anim = FuncAnimation(
    fig, update, frames=2000, interval=4, blit=True
)

plt.show()

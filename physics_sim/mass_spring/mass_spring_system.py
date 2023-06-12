import taichi as ti
import time
from physics_sim.solvers.solver import Solver
from physics_sim.constant import GRAVITY


@ti.data_oriented
class MassSpringSystem:
    def __init__(self,
                 current_num_particles: int = 5,
                 max_num_particles: int = 100,
                 mass: float = 1.0,
                 dt: float = 0.01,
                 solver: Solver = None,
                 spatial_dim: int = 2,
                 stiffness: float = 100,
                 spring_len: float = 0.2,
                 ground_height: float = 0.01,
                 fps: float = 8) -> None:
        self.current_num_particles = ti.field(ti.i32, shape=())
        self.current_num_particles[None] = current_num_particles
        self.max_num_particles = max_num_particles
        self.mass = mass
        self.dt = dt
        self.solver = solver
        self.spatial_dim = spatial_dim
        self.stiffness = stiffness
        self.spring_len = spring_len
        self.ground_height = ground_height
        self.fps = fps
        self.gui = ti.GUI("mass-spring",
                          res=(1024, 1024),
                          background_color=0xdddddd)
        self.g_coordinate = ti.Vector.field(n=self.spatial_dim,
                                            dtype=ti.f32,
                                            shape=[self.max_num_particles],
                                            needs_grad=True)
        self.g_velocity = ti.Vector.field(n=self.spatial_dim,
                                          dtype=ti.f32,
                                          shape=[self.max_num_particles])
        self.g_mass = ti.field(ti.f32, shape=[self.max_num_particles])
        self.v = ti.field(ti.f32, shape=(), needs_grad=True)
        self.force = ti.Vector.field(n=self.spatial_dim,
                                     dtype=ti.f32,
                                     shape=[self.max_num_particles])
        self.adjacent_field = ti.field(
            dtype=ti.f32,
            shape=[self.max_num_particles, self.max_num_particles])
        self.adjacent_distance = ti.field(
            dtype=ti.f32,
            shape=[self.max_num_particles, self.max_num_particles])

    def run(self):
        self.init()
        while self.gui.running:
            self.update_events()
            with ti.ad.Tape(loss=self.v):
                self.update_potential()
            self.update_force()
            self.update_dynamics()
            self.detect_collision()
            self.update_gui()
            self.gui.show()
            time.sleep(1. / self.fps)

    @ti.kernel
    def init(self):
        self.random_coordinate()
        for i in range(self.max_num_particles):
            self.g_mass[i] = self.mass

    def update_dynamics(self):
        # self.random_coordinate()
        self.solver.step(self.g_coordinate, self.g_velocity, self.g_mass, self.force, self.dt)
        self.update_adjacent()

    def update_gui(self):
        coordinates_ndarr = self.g_coordinate.to_numpy()
        print(coordinates_ndarr)
        self.gui.circles(coordinates_ndarr, radius=5.0, color=0xED553B)

        for i in range(self.current_num_particles[None]):
            for j in range(i, self.current_num_particles[None]):
                if self.adjacent_distance[i, j] > 0:
                    self.gui.line(begin=coordinates_ndarr[i],
                                  end=coordinates_ndarr[j],
                                  radius=2.0,
                                  color=10000)

    def update_events(self):
        if self.gui.get_event(ti.GUI.PRESS):
            if self.current_num_particles[None] > self.max_num_particles:
                self.gui.text(
                    f"ERROR, current particle num: {self.current_num_particles[None]} exceeds max particle num: {self.max_num_particles}",
                    pos=(0.5, 0.5))
            else:
                mouse_x, mouse_y = self.gui.get_cursor_pos()
                self.add_particles(mouse_x, mouse_y)
                self.current_num_particles[None] += 1

    @ti.kernel
    def update_adjacent(self):
        for i in range(self.current_num_particles[None]):
            self.adjacent_field[i, (i + 1) % self.max_num_particles] = ti.f32(1.0)
            self.adjacent_distance[i, (i + 1) % self.max_num_particles] = (
                self.g_coordinate[i] - self.g_coordinate[i + 1]).norm(1e-3)

    @ti.func
    def random_coordinate(self):
        for i in range(self.current_num_particles[None]):
            self.g_coordinate[i] = ti.Vector(
                [0.1 * ti.randn(ti.f32) + 0.5, 0.1 * ti.randn(ti.f32) + 0.5])

    @ti.kernel
    def add_particles(self, x: ti.f32, y: ti.f32):
        self.g_coordinate[self.current_num_particles[None]] = ti.Vector([x, y])
        self.g_velocity[self.current_num_particles[None]] = ti.Vector.zero(
            ti.f32, self.spatial_dim)

    @ti.kernel
    def update_potential(self):
        for i, j in ti.ndrange(self.current_num_particles[None], self.current_num_particles[None]):
            r = self.g_coordinate[i] - self.g_coordinate[j]
            self.v[None] += 0.5 * self.stiffness * self.adjacent_field[i, j] * ti.math.pow((r.norm(1e-3) - self.spring_len), 2)
        for i in range(self.current_num_particles[None]):
            self.v[None] += self.g_mass[i] * GRAVITY * self.g_coordinate[i][1]

    @ti.kernel
    def update_force(self):
        for i in ti.ndrange(self.current_num_particles[None]):
            self.force[i] = -self.g_coordinate.grad[i]

    @ti.kernel
    def detect_collision(self):
        for i in range(self.current_num_particles[None]):
            if self.g_coordinate[i][1] < self.ground_height:
                self.g_coordinate[i][1] = self.ground_height
                self.g_velocity[i][1] = 0.0


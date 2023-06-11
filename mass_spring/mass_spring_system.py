import taichi as ti
import time


@ti.data_oriented
class MassSpringSystem:
    def __init__(self,
                 current_num_particles: int = 5,
                 max_num_particles: int = 100,
                 spatial_dim: int = 2,
                 stiffness: float = 0.1,
                 fps: float = 8) -> None:
        self.current_num_particles = current_num_particles
        self.max_num_particles = max_num_particles
        self.spatial_dim = spatial_dim
        self.stiffness = stiffness
        self.fps = fps
        self.gui = ti.GUI("mass-spring",
                          res=(1024, 1024),
                          background_color=0xdddddd)
        self.g_coordinate = ti.Vector.field(n=self.spatial_dim,
                                            dtype=ti.f32,
                                            shape=[self.max_num_particles])
        self.g_velocity = ti.Vector.field(n=self.spatial_dim,
                                          dtype=ti.f32,
                                          shape=[self.max_num_particles])
        self.adjacent_field = ti.field(
            dtype=ti.i8,
            shape=[self.max_num_particles, self.max_num_particles])
        self.adjacent_distance = ti.field(
            dtype=ti.f32,
            shape=[self.max_num_particles, self.max_num_particles])

    def run(self):
        while self.gui.running:
            self.update_events()
            self.update_dynamics()
            self.update_gui()
            self.gui.show()
            time.sleep(1. / self.fps)

    @ti.kernel
    def update_dynamics(self):
        self.random_coordinate()
        self.update_adjacent()

    def update_gui(self):
        coordinates_ndarr = self.g_coordinate.to_numpy()
        self.gui.circles(coordinates_ndarr, radius=5.0, color=0xED553B)

        for i in range(self.current_num_particles):
            for j in range(i, self.current_num_particles):
                if self.adjacent_distance[i, j] > 0:
                    self.gui.line(begin=coordinates_ndarr[i],
                                  end=coordinates_ndarr[j],
                                  radius=2.0,
                                  color=10000)

    def update_events(self):
        if self.gui.get_event(ti.GUI.PRESS):
            if self.current_num_particles > self.max_num_particles:
                self.gui.text(
                    f"ERROR, current particle num: {self.current_num_particles} exceeds max particle num: {self.max_num_particles}",
                    pos=(0.5, 0.5))
            else:
                mouse_x, mouse_y = self.gui.get_cursor_pos()
                self.add_particles(mouse_x, mouse_y)
                self.current_num_particles += 1

    @ti.func
    def update_adjacent(self):
        for i in range(self.current_num_particles):
            self.adjacent_field[i, (i + 1) % self.max_num_particles] = ti.i8(1)
            self.adjacent_distance[i, (i + 1) % self.max_num_particles] = (
                self.g_coordinate[i] - self.g_coordinate[i + 1]).norm()

    @ti.func
    def random_coordinate(self):
        for i in range(self.current_num_particles):
            self.g_coordinate[i] = ti.Vector(
                [0.1 * ti.randn(ti.f32) + 0.5, 0.1 * ti.randn(ti.f32) + 0.5])

    @ti.kernel
    def add_particles(self, x: ti.f32, y: ti.f32):
        self.g_coordinate[self.current_num_particles] = ti.Vector([x, y])
        self.g_velocity[self.current_num_particles] = ti.Vector.zero(
            ti.f32, self.spatial_dim)


if __name__ == "__main__":
    ti.init(ti.gpu)
    mass_spring = MassSpringSystem()
    mass_spring.run()

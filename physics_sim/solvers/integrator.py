from abc import abstractmethod
import taichi as ti


@ti.data_oriented
class Integrator:
    def __init__(self) -> None:
        pass

    @abstractmethod
    @ti.kernel
    def step(self, coordinate: ti.template(), velocity: ti.template(),
             mass: ti.template(), force: ti.template(), dt: ti.template()):
        """
        """
        raise NotImplementedError


@ti.data_oriented
class ForwardEulerIntegrator(Integrator):
    def __init__(self) -> None:
        super().__init__()

    @ti.kernel
    def step(self, coordinate: ti.template(), velocity: ti.template(),
             mass: ti.template(), force: ti.template(), dt: ti.template()):
        for i in coordinate:
            coordinate[i] += velocity[i] * dt
            velocity[i] += force[i] / mass[i] * dt


@ti.data_oriented
class SymplecticIntegrator(Integrator):
    def __init__(self) -> None:
        super().__init__()

    @ti.kernel
    def step(self, coordinate: ti.template(), velocity: ti.template(),
             mass: ti.template(), force: ti.template(), dt: ti.template()):
        for i in coordinate:
            velocity[i] += force[i] / mass[i] * dt
            coordinate[i] += velocity[i] * dt


@ti.data_oriented
class ExplictIntegrator(Integrator):
    def __init__(self) -> None:
        super().__init__()

    # TODO
    @ti.kernel
    def step(self, coordinate: ti.template(), velocity: ti.template(),
             mass: ti.template(), force: ti.template(), dt: ti.template()):
        raise NotImplementedError
        # for i in coordinate:
        #     velocity[i] += force[i] / mass[i] * dt
        #     coordinate[i] += velocity[i] * dt


@ti.data_oriented
class RungeKuttaIntegrator(Integrator):
    def __init__(self, particle_num: int, spatial_dim: int) -> None:
        super().__init__()
        self.q1 = ti.Vector.field(spatial_dim, ti.f32, shape=particle_num)
        self.qdot1 = ti.Vector.field(spatial_dim, ti.f32, shape=particle_num)
        self.q2 = ti.Vector.field(spatial_dim, ti.f32, shape=particle_num)
        self.qdot2 = ti.Vector.field(spatial_dim, ti.f32, shape=particle_num)
        self.q3 = ti.Vector.field(spatial_dim, ti.f32, shape=particle_num)
        self.qdot3 = ti.Vector.field(spatial_dim, ti.f32, shape=particle_num)
        self.q4 = ti.Vector.field(spatial_dim, ti.f32, shape=particle_num)
        self.qdot4 = ti.Vector.field(spatial_dim, ti.f32, shape=particle_num)

    @ti.kernel
    def step(self, coordinate: ti.template(), velocity: ti.template(),
             mass: ti.template(), force: ti.template(), dt: ti.template()):
        small_dt = 1. / 6. * dt
        for i in coordinate:
            self.q1[i] = velocity[i] 
            self.qdot1 = 

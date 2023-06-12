import taichi as ti
from physics_sim.solvers.solver import Solver


@ti.data_oriented
class SymplecticSolver(Solver):
    def __init__(self) -> None:
        super().__init__()

    @ti.kernel
    def step(self, coordinate: ti.template(), velocity: ti.template(),
             mass: ti.template(), force: ti.template(), dt: ti.template()):
        for i in coordinate:
            velocity[i] += force[i] / mass[i] * dt
            coordinate[i] += velocity[i] * dt
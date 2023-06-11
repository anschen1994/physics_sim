import taichi as ti
from physics_sim.solvers.solver import Solver


@ti.data_oriented
class ForwardEulerSolver(Solver):
    def __init__(self) -> None:
        super().__init__()

    @ti.kernel
    def step(coordinate: ti.template(), velocity: ti.template(),
             mass: ti.template(), force: ti.template(), dt: ti.template()):
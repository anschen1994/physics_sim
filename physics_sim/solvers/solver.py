from abc import abstractmethod
import taichi as ti


@ti.data_oriented
class Solver:
    def __init__(self) -> None:
        pass

    @abstractmethod
    @ti.kernel
    def step(coordinate: ti.template(), velocity: ti.template(),
             mass: ti.template(), force: ti.template(), dt: ti.template()):
        """
        """
        raise NotImplementedError

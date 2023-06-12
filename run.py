import taichi as ti
import physics_sim.solvers.integrator as integrator
from physics_sim.mass_spring.mass_spring_system import MassSpringSystem





if __name__ == "__main__":
    ti.init(ti.cpu)
    # solver = integrator.ForwardEulerIntegrator()
    solver = integrator.SymplecticIntegrator()
    mass_spring = MassSpringSystem(current_num_particles=5, max_num_particles=10, solver=solver, stiffness=50, spring_len=0.4)
    mass_spring.run()
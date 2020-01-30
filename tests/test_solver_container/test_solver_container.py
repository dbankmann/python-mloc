from pymloc.solver_container import solver_container_factory
from pymloc.solvers import NullSolver
import pytest


class TestSolverContainer:
    def test_set_default_solver(self, local_opt):
        solver_container = solver_container_factory.get_solver_container(
            local_opt[0])
        solver_container.default_solver(NullSolver)
        assert solver_container.default_solver == NullSolver

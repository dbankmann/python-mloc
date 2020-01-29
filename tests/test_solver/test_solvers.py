from pymloc.solvers.solvers import SolverContainer, solver_container_factory
from pymloc.solvers.solver import NullSolver
from pymloc.model.optimization.local_optimization import LocalNullOptimization
import pytest


class TestSolvers:
    def test_set_default_solver(self):
        solver_container = solver_container_factory.get_solver_container(
            LocalNullOptimization)
        solver_container.default_solver(NullSolver)
        assert solver_container.default_solver == NullSolver


class TestSolverContainerFactory:
    def test_solver_container_creation(self):
        solver = NullSolver
        opt = LocalNullOptimization
        solver_container_factory.register_solver(solver, opt)

    @pytest.mark.xfail(reason="Not Implemented")
    def test_solver_subproblem_creation(self):
        assert False

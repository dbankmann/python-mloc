import pytest

from pymloc.model.optimization.local_optimization import LocalNullOptimization
from pymloc.model.solvable import solver_container_factory
from pymloc.solver_container import SolverContainer
from pymloc.solvers import NullSolver


class TestSolvers:
    def test_set_default_solver(self, local_opt):
        solver_container = solver_container_factory.get_solver_container(
            local_opt[0])
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

import pytest

from pymloc.model.optimization.local_optimization import LocalNullOptimization
from pymloc.solver_container import solver_container_factory
from pymloc.solvers.dynamical_systems.multiple_shooting import MultipleShooting


class TestSolverContainerFactory:
    def test_init(self):
        # Is only ever expected to fail, if test is run isolated without collecting other tests and thus imports
        assert len(solver_container_factory._solvers) > 0

    def test_solver_container_creation(self):
        from pymloc.solvers import NullSolver
        solver = NullSolver
        opt = LocalNullOptimization
        solver_container_factory.register_solver(solver, opt)

    def test_solver_subproblem_creation(self, initial_value_problem_dae):
        assert MultipleShooting in initial_value_problem_dae._available_solvers.solvers

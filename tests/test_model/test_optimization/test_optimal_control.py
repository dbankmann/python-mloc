import numpy as np
import pytest

from pymloc.model.control_system.dae import LinearControlSystem
from pymloc.model.optimization.optimal_control import LQOptimalControl
from pymloc.model.optimization.optimal_control import LQRConstraint
from pymloc.model.optimization.optimal_control import LQRObjective
from pymloc.model.variables.container import InputStateVariables
from pymloc.model.variables.time_function import Time

from .test_local_optimization import TestLocalOptimizationObject


@pytest.fixture
def dae_control():
    variables = InputStateVariables(1, 1)

    def e(t):
        return np.array([[1.]])

    def a(t):
        return np.array([[-1.]])

    def b(t):
        return np.array([[1.]])

    def c(t):
        return np.array([[1.]])

    def d(t):
        return np.array([[0.]])

    def f(t):
        return np.array([0.])

    return LinearControlSystem(variables, e, a, b, c, d, f)


@pytest.fixture
def lqr_obj_args():
    theta = 2.
    q = lambda t: np.array([[theta**2 - 1]])
    s = lambda t: np.array([[0.]])
    r = lambda t: np.array([[1.]])
    m = np.array([[0.]])
    time = Time(0., 10.)
    return time, q, s, r, m


class TestLQOptimalControl(TestLocalOptimizationObject):
    @pytest.fixture
    def loc_opt(self, dae_control, lqr_obj_args):
        variables = InputStateVariables(1, 1)
        x0 = np.array([2.])
        constraint = LQRConstraint(dae_control, x0)
        objective = LQRObjective(*lqr_obj_args)
        return LQOptimalControl(objective, constraint, variables)

    def test_init_solver(self, loc_opt):
        return super().test_init_solver(loc_opt, stepsize=1e-3)

    def test_solve(self, loc_opt):
        return super().test_solve(loc_opt, stepsize=1e-3)

    def test_solve2(self, loc_opt):
        time = Time(0., 10.)
        return super().test_solve2(loc_opt, time, stepsize=1e-3)

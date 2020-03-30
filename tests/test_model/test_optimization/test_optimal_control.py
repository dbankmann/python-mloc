import numpy as np
import pytest

from pymloc.model.control_system.dae import LinearControlSystem
from pymloc.model.optimization.optimal_control import LQOptimalControl
from pymloc.model.optimization.optimal_control import LQRConstraint
from pymloc.model.optimization.optimal_control import LQRObjective
from pymloc.model.variables.container import InputStateVariables
from pymloc.model.variables.time_function import Time

from .test_local_optimization import TestLocalOptimizationObject


def compute_ref_sol(theta, time, x0, t):
    tf = time.t_f
    t0 = time.t_0
    exp0 = np.exp(2 * theta * (tf - t0))
    exp1 = np.exp(-(t + t0) * theta)
    exp2 = np.exp(2 * t * theta)
    exp3 = np.exp(2 * tf * theta)
    tmp1 = theta + exp0 * (theta + 1) - 1
    tmp2 = np.array([
        -(exp2 - exp3) * (theta**2 - 1),
        (exp2 * (theta - 1) + exp3 * (theta + 1)),
        (exp2 - exp3) * (theta**2 - 1)
    ])

    refsol = tmp1**-1 * tmp2 * exp1 * x0
    return refsol


def compare_sol_ref_sol(loc_opt, theta=2.):
    loc_opt.init_solver(stepsize=1e-3, abs_tol=1e-12, rel_tol=1e-12)
    time = Time(0., 2.)
    time.grid = np.linspace(0., 2., 100)
    sol = loc_opt.solve(time, flow_abs_tol=1e-12, flow_rel_tol=1e-12)[0]
    x0 = loc_opt.constraint.initial_value
    for t in sol.time_grid:
        refsol = compute_ref_sol(theta, time, x0, t)
        assert np.allclose(sol(t), refsol)


@pytest.fixture
def dae_control():
    variables = InputStateVariables(1, 1, time=Time(0., 2.))

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
    time = Time(0., 2.)
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
        time = Time(0., 2.)
        sol = super().test_solve2(loc_opt, time, stepsize=1e-3)
        ref_sol = None

    def test_solve3(self, loc_opt):
        compare_sol_ref_sol(loc_opt)

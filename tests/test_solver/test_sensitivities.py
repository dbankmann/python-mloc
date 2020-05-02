#
# Copyright (c) 2019-2020
#
# @author: Daniel Bankmann
# @company: Technische Universität Berlin
#
# This file is part of the python package pymloc
# (see https://gitlab.tubit.tu-berlin.de/bankmann91/python-mloc )
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#
from copy import deepcopy

import jax.numpy as jnp
import numpy as np
import pytest

from pymloc.model.domains import RNDomain
from pymloc.model.dynamical_system.flow_problem import LinearFlow
from pymloc.model.dynamical_system.initial_value_problem import InitialValueProblem
from pymloc.model.dynamical_system.parameter_bvp import ParameterBoundaryValueProblem
from pymloc.model.dynamical_system.parameter_bvp import ParameterBoundaryValues
from pymloc.model.dynamical_system.parameter_dae import LinearParameterDAE
from pymloc.model.sensitivities.boundary_dae import BVPSensitivities
from pymloc.model.variables import NullVariables
from pymloc.model.variables import ParameterContainer
from pymloc.model.variables import StateVariablesContainer
from pymloc.model.variables.time_function import Time
from pymloc.solvers.dynamical_systems.adjoint_sensitivities import AdjointSensitivitiesSolver


@pytest.fixture
def bvp_sens_object(linear_param_bvp):
    return BVPSensitivities(linear_param_bvp, n_param=1)


@pytest.fixture
def sens_solver_sine(bvp_sens_object):
    return AdjointSensitivitiesSolver(bvp_sens_object,
                                      rel_tol=1e-1,
                                      abs_tol=1e-1)


@pytest.fixture
def sens_solver_sine2(bvp_sens_object):
    bvpsens2 = deepcopy(bvp_sens_object)
    bvpsens2.dynamical_system._a = lambda p, t: jnp.array([[0., 1.],
                                                           [-p**2, 0.]])
    bvpsens2.boundary_value_problem.boundary_values.inhomogeneity = lambda p: jnp.array(
        [0., 2.])

    return AdjointSensitivitiesSolver(bvpsens2, rel_tol=1e-6, abs_tol=1e-6)


@pytest.fixture
def sens_solver_sine3(bvp_sens_object):
    bvpsens2 = deepcopy(bvp_sens_object)
    bound_f = lambda p: jnp.array([[0., 0.], [p, 0.]])  # noqa: E731
    bvpsens2.boundary_value_problem.boundary_values.boundary_f = bound_f
    bvs = bvpsens2.boundary_value_problem.boundary_values.boundary_values
    bvs = list(bvs)
    bvs[1] = bound_f
    bvpsens2.boundary_value_problem.boundary_values._boundary_values = bvs

    bvpsens2.boundary_value_problem.boundary_values.inhomogeneity = lambda p: jnp.array(
        [0., 2.])  # noqa: E731
    return AdjointSensitivitiesSolver(bvpsens2, rel_tol=1e-1, abs_tol=1e-1)


@pytest.fixture
def linear_param_bvp(linear_param_dae, param_vars):
    initial_value = lambda p: jnp.array([[1., 0.], [0., 0.]])  # noqa: E731
    final_value = lambda p: jnp.array([[0., 0.], [1., 0.]])  # noqa: E731
    inhomogeneity = lambda p: jnp.array([0., p])  # noqa: E731
    t = Time(0., np.pi / 2., time_grid=np.linspace(0., np.pi / 2., 10))
    bvs = ParameterBoundaryValues(*param_vars, initial_value, final_value,
                                  inhomogeneity, 2)
    return ParameterBoundaryValueProblem(*param_vars, t, linear_param_dae, bvs)


@pytest.fixture
def linear_param_dae(param_vars, e_lin_param_dae, a_lin_param_dae,
                     f_lin_param_dae):
    return LinearParameterDAE(*param_vars, e_lin_param_dae, a_lin_param_dae,
                              f_lin_param_dae, 2)


@pytest.fixture
def param_vars():
    states = StateVariablesContainer(2)
    domain = RNDomain(1)
    parameters = ParameterContainer(1, domain)
    null_vars = NullVariables()
    return null_vars, parameters, states


@pytest.fixture
def e_lin_param_dae():
    def e(p, t):
        return np.identity(2)

    return e


@pytest.fixture
def a_lin_param_dae():
    def a(p, t):
        return jnp.array([[0., 1.], [-1., 0.]])

    return a


@pytest.fixture
def f_lin_param_dae():
    def f(p, t):
        return np.array([0., 0.])

    return f


class TestBVPSensitivitiesSolver:
    def test_compute_adjoint_boundary_values(self, sens_solver, localized_bvp):
        sens_solver._compute_adjoint_boundary_values(localized_bvp)

    def test_get_adjoint_dae_coeffs(self, sens_solver, localized_bvp):
        sens_solver._get_adjoint_dae_coeffs(localized_bvp)

    @pytest.mark.parametrize("tau", np.arange(0.1, 1., 0.4))
    def test_solve_adjoint_dae(self, sens_solver, localized_bvp, tau):
        parameters = localized_bvp._localization_parameters[0]
        time = deepcopy(localized_bvp.time_interval)
        stepsize = 0.5e-0
        time.grid = np.hstack(
            (np.arange(time.t_0, tau,
                       stepsize), np.arange(tau, time.t_f, stepsize)))
        sol = sens_solver._get_adjoint_solution(localized_bvp, parameters, tau,
                                                time)
        assert sol

    @pytest.mark.parametrize("tau", np.arange(0.1, 1., 0.4))
    def test_f_tilde(self, sens_solver, localized_bvp, tau):
        time = localized_bvp.time_interval
        flow_prob = LinearFlow(time, localized_bvp.dynamical_system)
        ivp_prob = InitialValueProblem(np.zeros((3, )), time,
                                       localized_bvp.dynamical_system)
        nodes = np.linspace(time.t_0, time.t_f, 3)
        stepsize = 0.5e-0
        localized_bvp.init_solver(flow_prob, ivp_prob, nodes, stepsize)
        sens_solver._get_capital_fs(localized_bvp,
                                    localized_bvp.solve(time)[0],
                                    localized_bvp._localization_parameters[0])

    @pytest.mark.parametrize("tau", np.arange(0.1, 1., 0.4))
    def test_run(self, sens_solver, localized_bvp, tau):
        sens_solver.run(localized_bvp._localization_parameters[0], tau)

    def test_sens_sine(self, sens_solver_sine):
        sol = sens_solver_sine.run(np.array(2.), 1.)
        assert np.allclose(sol.solution,
                           np.array([np.sin(1.), np.cos(1.)]),
                           atol=0.5,
                           rtol=1e-4)

    def test_sens_sine2(self, sens_solver_sine2):
        sol = sens_solver_sine2.run(np.array(1.), 1.)
        assert np.allclose(
            sol(1.).T,
            np.array([2 * np.cos(1.), 2 * np.cos(1.) - 2 * 1. * np.sin(1.)]),
            atol=1e-1,
            rtol=1e-9)

    def test_sens_sine3(self, sens_solver_sine3):
        sol = sens_solver_sine3.run(np.array(2.), 1.)
        assert np.allclose(sol.solution,
                           -2 * .25 *
                           np.array([np.sin(1.), np.cos(1.)]),
                           atol=0.5,
                           rtol=1e-7)

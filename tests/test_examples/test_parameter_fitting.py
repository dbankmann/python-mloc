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
import logging

import numpy as np
import pytest

import pymloc.model.optimization as optimization
from pymloc import MultiLevelOptimalControl
from pymloc.mloc import MultiLevelOptimalControl
from pymloc.model.optimization import NonLinearLeastSquares
from pymloc.model.variables import NullVariables
from pymloc.model.variables import Time

from ..test_model.test_optimization.test_optimal_control import compute_ref_sol
from ..test_model.test_parameter_dependent_optimal_control import refsol


@pytest.fixture
def f_nlsq():
    theta = 2.
    t0 = 0.
    tf = 2.
    x01 = 2.
    time = Time(t0, tf)
    t2 = 1.3
    t1 = 1.

    def f(ll_vars, hl_vars, loc_vars):
        sol1 = compute_ref_sol(theta, time, x01, t1)
        sol2 = compute_ref_sol(theta, time, x01, t2)
        sol0 = compute_ref_sol(theta, time, x01, t0)
        solf = compute_ref_sol(theta, time, x01, tf)
        f1 = ll_vars(t1) - sol1
        f2 = ll_vars(t2) - sol2
        f0 = ll_vars(t0) - sol0
        ff = ll_vars(tf) - solf
        return np.hstack((f0, f1, f2, ff))

    return f


@pytest.fixture
def nlsq_obj(f_nlsq, variables2):
    variables = (variables2[1], NullVariables(), variables2[0])
    return optimization.objectives.NonLinearLeastSquares(*variables, f_nlsq)


@pytest.fixture
def nlsq(nlsq_obj, variables2):
    variables = (variables2[1], NullVariables(), variables2[0])
    return NonLinearLeastSquares(nlsq_obj, *variables)


class TestParameterFitting:
    def test_mloc_solver_fail(self, pdoc_object, nlsq, variables2):
        optimizations = [nlsq, pdoc_object]
        solvers = [None, None]
        variables = (variables2[0], variables2[1])
        mloc = MultiLevelOptimalControl(optimizations, variables)
        with pytest.raises(ValueError):
            mloc.init_solver()

    def test_mloc_solver(self, pdoc_object, nlsq, variables2):
        optimizations = [nlsq, pdoc_object]
        solvers = [None, None]
        variables = (variables2[0], variables2[1])
        variables[0].current_values = np.array([1.])
        #TODO: Initialize correctly
        variables[1].current_values = np.array([])
        variables[1].time.grid = np.array([1., 1.3])
        logger = logging.getLogger("pymloc.solvers.nonlinear.gauss_newton")
        logger.setLevel(logging.DEBUG)
        mloc = MultiLevelOptimalControl(optimizations, variables)
        mloc.init_solver(abs_tol=1e-1, rel_tol=1e-1)
        solution = mloc.solve()
        assert np.allclose(solution.solution, 2., atol=1e-2)

    def test_mloc_solver_at_final(self, pdoc_object, nlsq, variables2):
        optimizations = [nlsq, pdoc_object]
        solvers = [None, None]
        variables = (variables2[0], variables2[1])
        variables[0].current_values = np.array([2.])
        #TODO: Initialize correctly
        variables[1].current_values = np.array([])
        variables[1].time.grid = np.array([1., 1.3])
        logger = logging.getLogger("pymloc.solvers.nonlinear.gauss_newton")
        logger.setLevel(logging.DEBUG)
        mloc = MultiLevelOptimalControl(optimizations, variables)
        mloc.init_solver(abs_tol=1e-1, rel_tol=1e-1)
        variables[0].associated_problem._solver_instance.upper_eta = 1.
        solution = mloc.solve()
        assert np.allclose(solution.solution, 2., atol=1e-12)

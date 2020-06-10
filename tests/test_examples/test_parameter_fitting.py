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
from pymloc.model.optimization import NonLinearLeastSquares
from pymloc.model.variables import NullVariables
from pymloc.model.variables import Time

from ..test_model.test_optimization.test_optimal_control import compute_ref_sol


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
def f_nlsq_3():
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
        f1 = ll_vars(t1)[1] - sol1[1]
        f2 = ll_vars(t2)[1] - sol2[1]
        f0 = ll_vars(t0)[1] - sol0[1]
        ff = ll_vars(tf)[1] - solf[1]
        return np.hstack((f0, f1, f2, ff))

    return f


def compute_ref_sol_augmented(theta, time, x01, t1):
    ref_sol = compute_ref_sol(theta, time, x01, t1)
    return np.array([ref_sol[0], 0., ref_sol[1], 0., ref_sol[2]])


@pytest.fixture
def f_nlsq_2():
    theta = 2.
    t0 = 0.
    tf = 2.
    x01 = 2.
    time = Time(t0, tf)
    t2 = 1.3
    t1 = 1.

    def f(ll_vars, hl_vars, loc_vars):
        sol1 = compute_ref_sol_augmented(theta, time, x01, t1)
        sol2 = compute_ref_sol_augmented(theta, time, x01, t2)
        sol0 = compute_ref_sol_augmented(theta, time, x01, t0)
        solf = compute_ref_sol_augmented(theta, time, x01, tf)
        f1 = ll_vars(t1)[2:] - sol1[2:]
        f2 = ll_vars(t2)[2:] - sol2[2:]
        f0 = ll_vars(t0)[2:] - sol0[2:]
        ff = ll_vars(tf)[2:] - solf[2:]
        return np.hstack((f0, f1, f2, ff))

    return f


@pytest.fixture
def nlsq_obj(f_nlsq, variables2):
    variables = (variables2[1], NullVariables(), variables2[0])
    return optimization.objectives.NonLinearLeastSquares(*variables, f_nlsq)


@pytest.fixture
def nlsq_obj_2(f_nlsq_2, variables3):
    variables = (variables3[1], NullVariables(), variables3[0])
    return optimization.objectives.NonLinearLeastSquares(*variables, f_nlsq_2)


@pytest.fixture
def nlsq_obj_3(f_nlsq_3, variables2):
    variables = (variables2[1], NullVariables(), variables2[0])
    return optimization.objectives.NonLinearLeastSquares(*variables, f_nlsq_3)


@pytest.fixture
def nlsq(nlsq_obj, variables2):
    variables = (variables2[1], NullVariables(), variables2[0])
    return NonLinearLeastSquares(nlsq_obj, *variables)


@pytest.fixture
def nlsq_3(nlsq_obj_3, variables2):
    variables = (variables2[1], NullVariables(), variables2[0])
    return NonLinearLeastSquares(nlsq_obj_3, *variables)


@pytest.fixture
def nlsq_2(nlsq_obj_2, variables3):
    variables = (variables3[1], NullVariables(), variables3[0])
    return NonLinearLeastSquares(nlsq_obj_2, *variables)


class TestParameterFitting:
    def test_mloc_solver_fail(self, pdoc_object, nlsq, variables2):
        optimizations = [nlsq, pdoc_object]
        variables = (variables2[0], variables2[1])
        mloc = MultiLevelOptimalControl(optimizations, variables)
        with pytest.raises(ValueError):
            mloc.init_solver()

    def test_mloc_solver(self, pdoc_object, nlsq_3, variables2):
        optimizations = [nlsq_3, pdoc_object]
        variables = (variables2[0], variables2[1])
        variables[0].current_values = np.array([1.])
        # TODO: Initialize correctly
        variables[1].current_values = np.array([])
        variables[1].time.grid = np.array([1., 1.3])
        pdoc_object.ll_sens_selector_shape = (1, 3)
        pdoc_object.ll_sens_selector = lambda p: np.array([[0., 1., 0.]])
        logger = logging.getLogger("pymloc.solvers.nonlinear.gauss_newton")
        logger.setLevel(logging.DEBUG)
        mloc = MultiLevelOptimalControl(optimizations, variables)
        mloc.init_solver(abs_tol=1e-1, rel_tol=1e-1)
        solution = mloc.solve()
        assert np.allclose(solution.solution, 2., atol=1e-1)

    def test_mloc_solver_precise(self, pdoc_object, nlsq_3, variables2):
        optimizations = [nlsq_3, pdoc_object]
        variables = (variables2[0], variables2[1])
        variables[0].current_values = np.array([1.])
        # TODO: Initialize correctly
        variables[1].current_values = np.array([])
        variables[1].time.grid = np.array([1., 1.3])
        pdoc_object.ll_sens_selector_shape = (1, 3)
        pdoc_object.ll_sens_selector = lambda p: np.array([[0., 1., 0.]])
        logger = logging.getLogger("pymloc.solvers.nonlinear.gauss_newton")
        logger.setLevel(logging.DEBUG)
        mloc = MultiLevelOptimalControl(optimizations, variables)

        np.set_printoptions(precision=8)
        mloc.init_solver(abs_tol=1e-6, rel_tol=1e-6)
        mloc.highest_opt.local_level_variables.associated_problem.solver_instance.upper_eta = 0.1
        solution = mloc.solve()
        logger.info("Solution: {}".format(solution.solution))
        assert np.allclose(solution.solution, 2., atol=9e-2)

    def test_mloc_solver_at_final(self, pdoc_object, nlsq, variables2):
        optimizations = [nlsq, pdoc_object]
        variables = (variables2[0], variables2[1])
        variables[0].current_values = np.array([2.])
        # TODO: Initialize correctly
        variables[1].current_values = np.array([])
        variables[1].time.grid = np.array([1., 1.3])
        logger = logging.getLogger("pymloc.solvers.nonlinear.gauss_newton")
        logger.setLevel(logging.DEBUG)
        mloc = MultiLevelOptimalControl(optimizations, variables)
        mloc.init_solver(abs_tol=1e-1, rel_tol=1e-1)
        variables[0].associated_problem._solver_instance.upper_eta = 1.
        solution = mloc.solve()
        assert np.allclose(solution.solution, 2., atol=1e-4)

    @pytest.mark.xfail
    def test_mloc_solver_2(self, pdoc_object_2, nlsq_2, variables3):
        optimizations = [nlsq_2, pdoc_object_2]
        variables = (variables3[0], variables3[1])
        variables[0].current_values = np.array([1., 2.])
        # TODO: Initialize correctly
        variables[1].current_values = np.array([])
        variables[1].time.grid = np.array([1., 1.3])
        logger = logging.getLogger("pymloc.solvers.nonlinear.gauss_newton")
        logger.setLevel(logging.DEBUG)
        pdoc_object_2.ll_sens_selector_shape = (3, 5)
        pdoc_object_2.ll_sens_selector = lambda p: np.block(
            [[np.zeros((2, 3))], [np.eye(3)]]).T
        # nlsq_2.ll_sens_selector_shape = (5, 3)
        # nlsq_2.ll_sens_selector = lambda p: np.linalg.block(
        #     [[np.zeros((2, 3))], [np.eye(3)]])
        mloc = MultiLevelOptimalControl(optimizations, variables)
        mloc.init_solver(abs_tol=1e-1, rel_tol=1e-1)
        solution = mloc.solve()
        assert np.allclose(solution.solution, 2., atol=1e-2)

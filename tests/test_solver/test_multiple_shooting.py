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
import numpy as np
import pytest
import scipy

from pymloc.misc import restack
from pymloc.misc import unstack
from pymloc.solvers.dynamical_systems.multiple_shooting import MultipleShooting
from pymloc.solvers.dynamical_systems.pygelda import PyGELDA


@pytest.fixture
def ms_object(initial_value_problem, flow_problem):
    stepsize = 1e-3
    interval = initial_value_problem.time_interval
    t0 = interval.t_0
    tf = interval.t_f
    interval.time_grid = np.linspace(t0, tf, 100)
    shooting_nodes = np.linspace(t0, tf, 3)
    ms_object = MultipleShooting(initial_value_problem,
                                 flow_problem,
                                 initial_value_problem,
                                 shooting_nodes,
                                 stepsize=stepsize)
    ms_object._init_solver(interval, flow_abs_tol=1e-6, flow_rel_tol=1e-6)
    return ms_object


@pytest.fixture
def ms_object_dae(initial_value_problem_dae, flow_problem_dae):
    stepsize = 1e-3
    interval = initial_value_problem_dae.time_interval
    t0 = interval.t_0
    tf = interval.t_f
    interval.time_grid = np.linspace(t0, tf, 100)
    shooting_nodes = np.linspace(t0, tf, 3)
    ms_object_dae = MultipleShooting(initial_value_problem_dae,
                                     flow_problem_dae,
                                     initial_value_problem_dae,
                                     shooting_nodes,
                                     stepsize=stepsize)
    ms_object_dae._init_solver(interval)
    return ms_object_dae


class TestMultipleShooting:
    def test_flow_matrix(self, ms_object):
        scipy_flow = scipy.linalg.expm(0.5 * ms_object._dynamical_system._a(0))
        assert np.allclose(ms_object._flows[:, :, 0], scipy_flow)

    def test_compute_t2(self, ms_object):
        nnodes = ms_object._n_shooting_nodes
        rank = ms_object._dynamical_system.rank
        nn = ms_object._nn
        assert ms_object._t2s.shape == (nn, rank, nnodes)

    def test_compute_gis(self, ms_object):
        gis = ms_object._gis

        nflows = ms_object._flows.shape[0]
        rank = ms_object._dynamical_system.rank
        assert gis.shape == (nflows, rank, rank)

    def test_compute_jis(self, ms_object):
        #rank initialization
        jis = ms_object._compute_jis()
        rank = ms_object._dynamical_system.rank
        nflows = ms_object._flows.shape[0]
        assert jis.shape == (nflows, rank, rank)

    def test_build_shooting_matrix(self, ms_object):
        ms_object._dynamical_system.init_rank()
        shooting_matrix = ms_object._get_shooting_matrix()
        rank = ms_object._dynamical_system.rank
        nflows = ms_object._n_shooting_nodes
        nn = ms_object._nn
        assert shooting_matrix.shape == (nflows * rank, nflows * rank)
        correct_result = np.array([[-0.22312958, 0., 1., 0., 0., 0.],
                                   [0., -0.60653091, 0., 1., 0., 0.],
                                   [0., 0., -0.22312958, 0., 1., 0.],
                                   [0., 0., 0., -0.60653091, 0., 1.],
                                   [1., 0., 0., 0., 0., 0.],
                                   [0., 1., 0., 0., 0., 0.]])
        assert np.allclose(shooting_matrix, correct_result)

    def test_newton_step(self, ms_object):
        initial_guess = ms_object._get_initial_guess()
        rhs = ms_object._get_newton_rhs(initial_guess)
        ms_object._newton_step(initial_guess, rhs)

    def test_get_newton_rhs(self, ms_object):
        ms_object._get_shooting_values()
        states = np.random.random((2, 3))
        rhs = ms_object._get_newton_rhs(states)
        assert rhs.shape == (6, )

    def test_run(self, ms_object):
        time_interval = ms_object._ivp_problem.time_interval
        erg = ms_object.run(time_interval,
                            flow_abs_tol=1e-6,
                            flow_rel_tol=1e-6)[1]
        scipy_flow = scipy.linalg.expm(0.5 * ms_object._dynamical_system._a(0))
        iv = ms_object._bvp.initial_value
        comp = np.block([iv, scipy_flow @ iv,
                         scipy_flow @ scipy_flow @ iv]).reshape(2,
                                                                3,
                                                                order='F')
        assert np.allclose(erg.solution, comp)

    def test_run_dae(self, ms_object_dae):
        time_interval = ms_object_dae._ivp_problem.time_interval
        erg = ms_object_dae.run(time_interval)[1]

    def test_unstack(self, rand_shape):
        a = rand_shape
        b = unstack(a)
        np.testing.assert_equal(a[..., 0], b[:a.shape[0], ...])

    def test_restack(self, rand_shape):
        b = unstack(rand_shape)
        np.testing.assert_equal(restack(b, rand_shape.shape), rand_shape)

    @pytest.fixture(params=np.arange(100))
    def rand_shape(self):
        dim = np.random.randint(2, 9)
        shape = list(np.random.randint(1, 10, dim))
        size = np.empty(shape).size
        return np.arange(size).reshape(shape)

    def test_intermediate(self, ms_object):
        time_interval = ms_object._ivp_problem.time_interval
        time_interval.time_grid = np.linspace(0, 1, 3)
        erg, node_erg = ms_object.run(time_interval)
        for node in node_erg.time_grid:
            assert np.allclose(node_erg(node), erg(node), atol=1e-6)

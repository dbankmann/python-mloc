import numpy as np
import pytest
import scipy

from pymloc.solvers.dynamical_systems.multiple_shooting import MultipleShooting
from pymloc.solvers.dynamical_systems.pygelda import PyGELDA


@pytest.fixture
def ms_object(initial_value_problem, flow_problem):
    stepsize = 1e-3
    interval = initial_value_problem.time_interval
    t0 = interval.t_0
    tf = interval.t_f
    shooting_nodes = np.linspace(t0, tf, 3)
    ms_object = MultipleShooting(initial_value_problem,
                                 flow_problem,
                                 shooting_nodes,
                                 stepsize=stepsize)
    ms_object._init_solver()
    return ms_object


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
        gis = ms_object._compute_gis()

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
        erg = ms_object.run()
        scipy_flow = scipy.linalg.expm(0.5 * ms_object._dynamical_system._a(0))
        iv = ms_object._bvp.initial_value
        comp = np.block([iv, scipy_flow @ iv,
                         scipy_flow @ scipy_flow @ iv]).reshape(2,
                                                                3,
                                                                order='F')
        assert np.allclose(erg, comp)

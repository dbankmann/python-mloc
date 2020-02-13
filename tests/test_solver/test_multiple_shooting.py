import numpy as np
import pytest
import scipy

from pymloc.solvers.dynamical_systems.multiple_shooting import MultipleShooting
from pymloc.solvers.dynamical_systems.pygelda import PyGELDA


@pytest.fixture
def ms_object(initial_value_problem):
    stepsize = 1e-3
    initial_value_problem.dynamical_system._f = lambda t: np.array([0., 0.])
    pygelda = PyGELDA(initial_value_problem, stepsize)
    interval = initial_value_problem.time_interval
    t0 = interval.t_0
    tf = interval.t_f
    shooting_nodes = np.linspace(t0, tf, 2)
    return MultipleShooting(initial_value_problem,
                            pygelda,
                            shooting_nodes,
                            stepsize=stepsize)


class TestMultipleShooting:
    def test_flow_matrix(self, ms_object):
        flows = ms_object._compute_flows()
        scipy_flow = scipy.linalg.expm(ms_object._dynamical_system._a(0))
        assert np.allclose(flows[:, :, 0], scipy_flow)

    def test_compute_t2(self, ms_object):
        t2s = ms_object._compute_t2()
        nnodes = ms_object._n_shooting_nodes
        rank = ms_object._dynamical_system._current_rank
        nn = ms_object._nn
        assert np.array(t2s).shape == (nn, rank, nnodes)

    def test_compute_inner_t2(self, ms_object):
        t2s = ms_object._compute_t2()
        inner_t2s = ms_object._compute_inner_t2_tilde(t2s)
        rank = ms_object._dynamical_system._current_rank
        nn = ms_object._nn
        assert ms_object._n_shooting_nodes == len(inner_t2s) + 2

    def test_compute_gi(self, ms_object):
        ms_object._compute_gi

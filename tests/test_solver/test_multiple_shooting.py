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
    shooting_nodes = np.linspace(t0, tf, 3)
    return MultipleShooting(initial_value_problem,
                            pygelda,
                            shooting_nodes,
                            stepsize=stepsize)


@pytest.fixture
def flows(ms_object):
    return ms_object._compute_flows()


@pytest.fixture
def t2s(ms_object):
    return ms_object._compute_t2()


class TestMultipleShooting:
    def test_flow_matrix(self, ms_object, flows):
        scipy_flow = scipy.linalg.expm(0.5 * ms_object._dynamical_system._a(0))
        assert np.allclose(flows[0, :, :], scipy_flow)

    def test_compute_t2(self, ms_object, t2s):
        nnodes = ms_object._n_shooting_nodes
        rank = ms_object._dynamical_system.rank
        nn = ms_object._nn
        assert t2s.shape == (nnodes, nn, rank)

    def test_compute_gis(self, ms_object, t2s, flows):
        gis = ms_object._compute_gis(t2s, flows)

        nflows = flows.shape[0]
        rank = ms_object._dynamical_system.rank
        assert gis.shape == (nflows, rank, rank)

    def test_compute_jis(self, ms_object, flows):
        #rank initialization
        ms_object._compute_t2()
        jis = ms_object._compute_jis()
        rank = ms_object._dynamical_system.rank
        nflows = ms_object._n_shooting_nodes
        assert jis.shape == (nflows, rank, rank)

    def test_build_shooting_matrix(self, ms_object):
        shooting_matrix = ms_object._build_shooting_matrix()
        rank = ms_object._dynamical_system.rank
        nflows = ms_object._n_shooting_nodes
        nn = ms_object._nn
        assert shooting_matrix.shape == (nflows * rank + nn, nflows * rank)

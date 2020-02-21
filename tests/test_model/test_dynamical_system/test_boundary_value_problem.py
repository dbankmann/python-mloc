import numpy as np
import pytest

from pymloc.model.dynamical_system.boundary_value_problem import \
    BoundaryValueProblem
from pymloc.model.variables.time_function import Time


@pytest.fixture
def bvp(linear_dae):
    gamma_0 = np.diag((1., 2.))
    gamma_f = np.diag((0., 1.))
    gamma = np.array([1., 3.])
    time_interval = Time(0., 1.)
    return BoundaryValueProblem(time_interval, gamma_0, gamma_f, gamma,
                                linear_dae)


class TestBoundaryValueProblem:
    def test_init(self, bvp):
        assert bvp._nnodes == 2

    def test_boundary_residual(self, bvp):
        node_values = np.array([[10., 100.], [1000., -10.]])
        residual = bvp.boundary_residual(node_values)
        assert np.allclose(
            residual,
            np.array([10. + 0., 2000. + -10.]) - np.array([1., 3.]))

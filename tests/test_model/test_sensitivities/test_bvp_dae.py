import numpy as np
import pytest

from pymloc.model.dynamical_system.boundary_value_problem import MultipleBoundaryValueProblem
from pymloc.model.dynamical_system.flow_problem import LinearFlow
from pymloc.model.sensitivities.boundary_dae import BVPSensitivities
from pymloc.model.variables.time_function import Time


@pytest.fixture
def bvp_sens_object(linear_param_bvp):
    return BVPSensitivities(linear_param_bvp)


@pytest.fixture(params=np.arange(0.1, 2., 0.1))
def localized_bvp(bvp_sens_object, request):
    parameters = np.array([request.param])
    return bvp_sens_object.get_sensitivity_bvp(parameters)


@pytest.fixture
def localized_flow_prob(localized_bvp):
    time_interval = Time(0., 1.)
    flowprob = LinearFlow(time_interval, localized_bvp.dynamical_system)
    return flowprob


class TestBVPSensitivities:
    def test_get_sens_bvp(self, bvp_sens_object):
        parameters = np.array([2.])
        assert isinstance(bvp_sens_object.get_sensitivity_bvp(parameters),
                          MultipleBoundaryValueProblem)

    def test_compute_adjoint_boundary_values(self, bvp_sens_object,
                                             localized_bvp):
        bvp_sens_object._compute_adjoint_boundary_values(localized_bvp)

    def test_solve_localized(self, localized_bvp, localized_flow_prob):
        nodes = np.linspace(0, 1, 5)
        stepsize = 1e-4
        localized_bvp.init_solver(localized_flow_prob, nodes, stepsize)
        localized_bvp.solve()

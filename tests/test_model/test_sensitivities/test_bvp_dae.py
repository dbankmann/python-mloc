import numpy as np
import pytest

from pymloc.model.dynamical_system.boundary_value_problem import MultipleBoundaryValueProblem
from pymloc.model.dynamical_system.flow_problem import LinearFlow
from pymloc.model.sensitivities.boundary_dae import BVPSensitivities
from pymloc.model.variables.time_function import Time


@pytest.fixture
def bvp_sens_object(linear_param_bvp):
    return BVPSensitivities(linear_param_bvp, 1)


class TestBVPSensitivities:
    def test_get_sens_bvp(self, bvp_sens_object):
        parameters = np.array([2.])
        assert isinstance(bvp_sens_object.get_sensitivity_bvp(parameters),
                          MultipleBoundaryValueProblem)

    def test_solve_localized(self, localized_bvp, localized_ivp,
                             localized_flow_prob):
        nodes = np.linspace(0, 1, 3)
        stepsize = 1e-4
        time_interval = Time(0., 1., time_grid=np.linspace(0., 1., 10))

        localized_bvp.init_solver(localized_flow_prob, localized_ivp, nodes,
                                  stepsize)
        localized_bvp.solve(time_interval)

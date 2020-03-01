import numpy as np

from ..dynamical_system.parameter_bvp import ParameterBoundaryValueProblem
from ..solvable import Solvable


class BVPSensitivities(Solvable):
    def __init__(self,
                 boundary_value_problem: ParameterBoundaryValueProblem,
                 n_param,
                 selector=None):
        super().__init__()
        if not isinstance(boundary_value_problem,
                          ParameterBoundaryValueProblem):
            raise TypeError(
                "Only ParameterBoundaryValueProblem is supported at the moment"
            )
        self._bvp = boundary_value_problem
        self._dynamical_system = boundary_value_problem.dynamical_system
        self._time_interval = self._bvp.time_interval
        self._n_param = n_param
        if selector is None:
            selector = np.identity(self._dynamical_system.nn)
        self._selector = selector

    @property
    def n_param(self):
        return self._n_param

    @property
    def boundary_value_problem(self):
        return self._bvp

    @property
    def dynamical_system(self):
        return self._dynamical_system

    @property
    def selector(self):
        return self._selector

    def get_sensitivity_bvp(self, parameters):
        return self._bvp.get_localized_object(parameters=parameters)

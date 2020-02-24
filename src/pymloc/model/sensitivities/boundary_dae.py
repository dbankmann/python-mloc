import numpy as np
import scipy.linalg as linalg

from ..dynamical_system.parameter_bvp import ParameterBoundaryValueProblem
from ..solvable import Solvable


class BVPSensitivities(Solvable):
    def __init__(self, boundary_value_problem: ParameterBoundaryValueProblem):
        super().__init__()
        if not isinstance(boundary_value_problem,
                          ParameterBoundaryValueProblem):
            raise TypeError(
                "Only ParameterBoundaryValueProblem is supported at the moment"
            )
        self._bvp = boundary_value_problem
        self._dynamical_system = boundary_value_problem.dynamical_system
        self._time_interval = self._bvp.time_interval

    def get_sensitivity_bvp(self, parameters):
        return self._bvp.get_localized_object(parameters=parameters)

    def _compute_adjoint_boundary_values(self, localized_bvp):
        n = self._bvp.dynamical_system.nn
        rank = localized_bvp.dynamical_system.rank
        t_0 = self._time_interval.t_0
        t_f = self._time_interval.t_f
        t20 = localized_bvp.dynamical_system.t2(t_0)
        t2f = localized_bvp.dynamical_system.t2(t_f)
        z_gamma = localized_bvp.boundary_values.z_gamma
        gamma_0, gamma_f = localized_bvp.boundary_values.boundary_values.transpose(
            2, 0, 1)
        temp = z_gamma.T @ np.block([gamma_0 @ t20, gamma_f @ t2f])
        small_new_gammas = linalg.null_space(temp)
        gamma_check_0 = z_gamma @ small_new_gammas[:rank, :].T @ t20.T
        gamma_check_f = z_gamma @ small_new_gammas[rank:, :].T @ t2f.T

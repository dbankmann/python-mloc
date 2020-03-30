import logging
from copy import deepcopy

import numpy as np
import scipy
import scipy.linalg as linalg

from pymloc.model.dynamical_system.flow_problem import LinearFlow

from ...model.dynamical_system.boundary_value_problem import BoundaryValueProblem
from ...model.dynamical_system.boundary_value_problem import BoundaryValues
from ...model.dynamical_system.boundary_value_problem import MultipleBoundaryValueProblem
from ...model.dynamical_system.boundary_value_problem import MultipleBoundaryValues
from ...model.dynamical_system.initial_value_problem import InitialValueProblem
from ...model.dynamical_system.representations import LinearFlowRepresentation
from ...model.sensitivities.boundary_dae import BVPSensitivities
from ...model.variables.container import StateVariablesContainer
from ...model.variables.time_function import Time
from ...solver_container import solver_container_factory
from ..base_solver import BaseSolver
from ..base_solver import TimeSolution


class SensitivitiesSolver(BaseSolver):
    def __init__(self, bvp_param, *args, **kwargs):
        if not isinstance(bvp_param, BVPSensitivities):
            raise TypeError(bvp_param)
        self._bvp_param = bvp_param
        self._dynamical_system = bvp_param.dynamical_system
        self._nn = self._dynamical_system.nn
        self._time_interval = self._bvp_param.time_interval
        self._boundary_values = self._bvp_param.boundary_value_problem.boundary_values
        super().__init__(*args, **kwargs)

    def _get_capital_f_tilde(self, localized_bvp, solution, parameter):
        def a_dif(t):
            return self._dynamical_system.a_theta(parameter, t)

        def e_dif(t):
            return self._dynamical_system.e_theta(parameter, t)

        def f_dif(t):
            return self._dynamical_system.f_theta(parameter, t)

        def x_d(t):
            return localized_bvp.dynamical_system.x_d(t, solution(t))

        def x_d_dot(t):
            return np.einsum('ij,j->i', localized_bvp.dynamical_system.d_d(t),
                             x_d(t))

        def f_tilde(t):
            f_tilde = np.einsum(
                'ijk,j->ik', a_dif(t), solution(t)) - np.einsum(
                    'ijk,j->ik', e_dif(t), x_d_dot(t)) + f_dif(t)
            return f_tilde  #TODO: Asssumes that Si === 0 in Remark 8 of thesis for forward sensitivities.

        return f_tilde

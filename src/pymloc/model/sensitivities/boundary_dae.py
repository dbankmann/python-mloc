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

from ..dynamical_system.parameter_bvp import ParameterBoundaryValueProblem
from ..dynamical_system.parameter_dae import jac_jax_reshaped
from ..solvable import Solvable


class BVPSensitivities(Solvable):
    def __init__(self,
                 boundary_value_problem: ParameterBoundaryValueProblem,
                 n_param,
                 selector=None,
                 selector_shape=None):
        super().__init__()
        if not isinstance(boundary_value_problem,
                          ParameterBoundaryValueProblem):
            raise TypeError(
                "Only ParameterBoundaryValueProblem is supported at the moment"
            )
        self._bvp = boundary_value_problem
        self._dynamical_system = boundary_value_problem.dynamical_system
        self._time_interval = self._bvp.time_interval
        self._parameters = self._bvp.higher_level_variables
        self._n_param = n_param
        if selector is None:
            selector = lambda p: np.identity(self._dynamical_system.nn)
            self._sel_shape = (self._dynamical_system.nn,
                               self._dynamical_system.nn)
        else:
            self._sel_shape = selector_shape
        self._selector = selector

    @property
    def parameters(self):
        return self._parameters

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

    @property
    def selector_theta(self):
        shape = self._sel_shape
        return jac_jax_reshaped(self.selector, shape)

    @property
    def time_interval(self):
        return self._time_interval

    def get_sensitivity_bvp(self, parameters=None):
        return self._bvp.get_localized_object(parameters=parameters)

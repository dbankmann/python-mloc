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

from ..multilevel_object import local_object_factory
from .parameter_bvp import AutomaticMultipleBoundaryValueProblem
from .parameter_bvp import ParameterBoundaryValueProblem
from .parameter_bvp import ParameterBoundaryValues


class ParameterInitialValueProblem(ParameterBoundaryValueProblem):
    def __init__(self,
                 ll_vars,
                 hl_vars,
                 loc_vars,
                 initial_value,
                 time_interval,
                 dynamical_system,
                 n_param=1):
        self._initial_value = initial_value
        n = dynamical_system.nn

        def boundary_0(p):
            return np.identity(n)

        def boundary_f(p):
            return np.zeros((n, n))

        bound_values = ParameterBoundaryValues(ll_vars, hl_vars, loc_vars,
                                               boundary_0, boundary_f,
                                               initial_value, n, n_param)
        super().__init__(ll_vars, hl_vars, loc_vars, time_interval,
                         dynamical_system, bound_values)

    @property
    def initial_value(self):
        return self._initial_value


local_object_factory.register_localizer(ParameterInitialValueProblem,
                                        AutomaticMultipleBoundaryValueProblem)

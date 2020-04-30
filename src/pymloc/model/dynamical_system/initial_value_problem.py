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
from abc import ABC

import numpy as np

from .boundary_value_problem import BoundaryValueProblem
from .boundary_value_problem import BoundaryValues


class InitialValueProblem(BoundaryValueProblem):
    def __init__(self, initial_value, time_interval, dynamical_system):
        self._initial_value = initial_value
        n = initial_value.size
        bound_values = BoundaryValues(np.identity(n), np.zeros((n, n)),
                                      initial_value)
        super().__init__(time_interval, dynamical_system, bound_values)

    @property
    def initial_value(self):
        return self._initial_value

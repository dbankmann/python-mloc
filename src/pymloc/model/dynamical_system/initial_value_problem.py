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

from ..variables import Time
from .boundary_value_problem import BoundaryValueProblem
from .boundary_value_problem import BoundaryValues


class InitialValueProblem(BoundaryValueProblem):
    """Defines initial value problems for dynamical systems."""
    def __init__(self, initial_value: np.ndarray, time_interval: Time,
                 dynamical_system):
        self._initial_value = initial_value
        n = initial_value.size
        bound_values = BoundaryValues(np.identity(n), np.zeros((n, n)),
                                      initial_value)
        super().__init__(time_interval, dynamical_system, bound_values)

    @property
    def initial_value(self) -> np.ndarray:
        """The initial value of the initial value problem."""
        return self._initial_value

from .boundary_value_problem import BoundaryValueProblem
from abc import ABC
import numpy as np


class InitialValueProblem(BoundaryValueProblem):
    def __init__(self, initial_value, time_interval, dynamical_system):
        self._initial_value = initial_value
        n = initial_value.size
        super().__init__(time_interval, np.identity(n), np.zeros((n, n)),
                         initial_value, dynamical_system)

    @property
    def initial_value(self):
        return self._initial_value

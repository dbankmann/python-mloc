from ..solvable import Solvable
from abc import ABC

import numpy as np


class MultipleBoundaryValueProblem(Solvable, ABC):
    def __init__(self, boundary_values, inhomogeinity, dynamical_system):
        self._boundary_values = boundary_values
        self._inhomogeinity = inhomogeinity
        self._dynamical_system = dynamical_system


class BoundaryValueProblem(MultipleBoundaryValueProblem):
    def __init__(self, boundary_0, boundary_f, inhomogeneity,
                 dynamical_system):
        self.boundary_0 = boundary_0
        self.boundary_f = boundary_f
        self.inhomogeinity = inhomogeneity
        super().__init__((
            boundary_0,
            boundary_f,
        ), inhomogeneity, dynamical_system)


class InitialValueProblem(BoundaryValueProblem):
    def __init__(self, initial_value, dynamical_system):
        self._initial_value = initial_value
        n = initial_value.size
        super().__init__(np.identity(n), np.zeros((n, n)), initial_value,
                         dynamical_system)

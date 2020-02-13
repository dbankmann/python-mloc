from abc import ABC

import numpy as np

from ..solvable import Solvable


class MultipleBoundaryValueProblem(Solvable, ABC):
    def __init__(self, timepoints, boundary_values, inhomogeinity,
                 dynamical_system):
        self._boundary_values = boundary_values
        self._timepoints = timepoints
        self._nnodes = len(boundary_values)
        if len(timepoints) != self._nnodes:
            raise ValueError(
                "Timepoints and boundary values need to have same size.")
        self._inner_nodes = boundary_values[1:-1]
        self._inhomogeinity = inhomogeinity
        self._dynamical_system = dynamical_system
        self.nn = dynamical_system.nn
        self.nm = dynamical_system.nm
        super().__init__()

    @property
    def dynamical_system(self):
        return self._dynamical_system


class BoundaryValueProblem(MultipleBoundaryValueProblem):
    def __init__(self, time_interval, boundary_0, boundary_f, inhomogeneity,
                 dynamical_system):
        self.boundary_0 = boundary_0
        self.boundary_f = boundary_f
        self._time_interval = time_interval
        self._initial_time = time_interval.t_0
        self._final_time = time_interval.t_f
        self.inhomogeinity = inhomogeneity
        super().__init__((time_interval.t_0, time_interval.t_f), (
            boundary_0,
            boundary_f,
        ), inhomogeneity, dynamical_system)

    @property
    def initial_time(self):
        return self._initial_time

    @property
    def final_time(self):
        return self._final_time

    @property
    def time_interval(self):
        return self._time_interval

    @property
    def boundary_values(self):
        return self._boundary_values

    @property
    def time_points(self):
        return self._timepoints

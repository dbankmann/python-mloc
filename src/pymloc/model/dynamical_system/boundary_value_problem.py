from abc import ABC

import numpy as np

from ..solvable import Solvable


class MultipleBoundaryValueProblem(Solvable, ABC):
    def __init__(self,
                 timepoints,
                 boundary_values,
                 inhomogeinity,
                 dynamical_system,
                 z_gamma=None):
        self._nnodes = len(boundary_values)
        if len(timepoints) != self._nnodes:
            raise ValueError(
                "Timepoints and boundary values need to have same size.")
        self._boundary_values = self._set_bvs(boundary_values)
        self._timepoints = timepoints
        self._inner_nodes = boundary_values[1:-1]
        self._inhomogeinity = inhomogeinity
        self._dynamical_system = dynamical_system
        self.nn = dynamical_system.nn
        self.nm = dynamical_system.nm
        self._z_gamma = self._set_z_gamma(z_gamma)
        super().__init__()

    def _set_z_gamma(self, z_gamma):
        if z_gamma is None:
            rank = self.dynamical_system.rank
            z_gamma = np.zeros((rank, self.nn), order='F')
            z_gamma[:rank, :rank] = np.identity(rank)
        return z_gamma

    @property
    def z_gamma(self):
        return self._z_gamma

    def _set_bvs(self, bvs):
        return np.array(list(bv.T for bv in bvs)).T

    @property
    def dynamical_system(self):
        return self._dynamical_system

    def boundary_residual(self, node_values):
        #TODO: Make more efficient (save intermediate products)
        residual = np.einsum('hi,ijk,jk->h', self._z_gamma,
                             self._boundary_values,
                             node_values) - self._z_gamma @ self._inhomogeinity
        return residual


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

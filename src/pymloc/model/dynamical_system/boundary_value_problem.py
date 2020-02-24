from abc import ABC

import numpy as np

from ..solvable import Solvable


class MultipleBoundaryValues:
    def __init__(self, boundary_values, inhomogeinity, z_gamma=None):
        self._nnodes = len(boundary_values)
        self._boundary_values = self._set_bvs(boundary_values)
        self._inner_nodes = boundary_values[1:-1]
        self._inhomogeinity = inhomogeinity
        self._z_gamma = z_gamma

    def residual(self, node_values):
        #TODO: Make more efficient (save intermediate products)
        residual = np.einsum('hi,ijk,jk->h', self._z_gamma,
                             self._boundary_values,
                             node_values) - self._z_gamma @ self._inhomogeinity
        return residual

    @property
    def nnodes(self):
        return self._nnodes

    @property
    def boundary_values(self):
        return self._boundary_values

    @property
    def inhomogeinity(self):
        return self._inhomogeinity

    @property
    def z_gamma(self):
        return self._z_gamma

    def set_z_gamma(self, rank, n):
        if self.z_gamma is None:
            z_gamma = np.zeros((rank, n), order='F')
            z_gamma[:rank, :rank] = np.identity(rank)
        self._z_gamma = z_gamma

    def _set_bvs(self, bvs):
        return np.array(list(bv.T for bv in bvs)).T


class BoundaryValues(MultipleBoundaryValues):
    def __init__(self, boundary_0, boundary_f, inhomogeneity, z_gamma=None):
        self.boundary_0 = boundary_0
        self.boundary_f = boundary_f
        super().__init__((
            boundary_0,
            boundary_f,
        ), inhomogeneity, z_gamma)


class MultipleBoundaryValueProblem(Solvable, ABC):
    def __init__(self, time_intervals, dynamical_system, boundary_values):
        self._time_intervals = time_intervals
        self._dynamical_system = dynamical_system
        self.nn = dynamical_system.nn
        self.nm = dynamical_system.nm
        self._boundary_values = boundary_values
        boundary_values.set_z_gamma(dynamical_system.rank, self.nn)
        if len(time_intervals) + 1 != self._boundary_values.nnodes:
            raise ValueError(
                "Timepoints and boundary values need to have same size.")
        self._nodes = self._get_and_check_nodes()
        super().__init__()

    @property
    def dynamical_system(self):
        return self._dynamical_system

    @property
    def boundary_values(self):
        return self._boundary_values

    @property
    def time_points(self):
        return self._timepoints

    def _get_and_check_nodes(self):
        nodes = ()
        tis = self._time_intervals
        nodes += (tis[0].t_0, )
        for first_i, second_i in zip(tis, tis[1:]):
            tf = first_i.t_f
            t0 = second_i.t_0
            if not np.close(t0, tf):
                raise ValueError("TimeIntervals must intersect at boundary")
            nodes += (tf, )
        nodes += (tis[-1].t_f, )
        return nodes

    @property
    def nodes(self):
        return self._nodes


class BoundaryValueProblem(MultipleBoundaryValueProblem):
    def __init__(self, time_interval, dynamical_system,
                 boundary_values: BoundaryValues):
        self._time_interval = time_interval
        self._initial_time = time_interval.t_0
        self._final_time = time_interval.t_f
        super().__init__((time_interval, ), dynamical_system, boundary_values)

    @property
    def initial_time(self):
        return self._initial_time

    @property
    def final_time(self):
        return self._final_time

    @property
    def time_interval(self):
        return self._time_interval

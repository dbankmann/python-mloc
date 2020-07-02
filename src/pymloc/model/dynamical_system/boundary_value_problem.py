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

from typing import Optional
from typing import Tuple

import jax.numpy as jnp
import numpy as np

import pymloc

from ..solvable import VariableSolvable
from ..variables.time_function import Time


class MultipleBoundaryValues:
    r"""Base class for all boundary values of a bounday value problem of the form :class:`.MultipleBoundaryValueProblem`
    The boundary conditions have the general form

    .. math::
        \sum_{i=0}^{q}{\Gamma_{i}x(t_{i})}=\gamma.
    """
    def __init__(self,
                 boundary_coefficients: Tuple[np.ndarray, ...],
                 inhomogeneity: np.ndarray,
                 z_gamma: Optional[np.ndarray] = None):
        """

        Parameters
        ----------
        boundary_coefficients: Boundary coefficients for every corresponding boundary node
        inhomogeneity: righthandside of the boundary condition
        z_gamma: Selector matrix of the boundary condition
        """
        self._nnodes: int = len(boundary_coefficients)
        self._boundary_values: np.ndarray = self._set_bvs(
            boundary_coefficients)
        self._inner_nodes: np.ndarray = boundary_coefficients[1:-1]
        # TODO: Check size of inhomogeneity according to state dimension (vector or matrix)
        self._inhomogeneity: np.ndarray = inhomogeneity
        self._z_gamma: Optional[np.ndarray] = z_gamma
        # Get last dimension, if only a vector, set to1
        self._n_inhom: int = np.atleast_2d(inhomogeneity.T).T.shape[-1]

    def residual(self, node_values: np.ndarray) -> np.ndarray:
        """Computes the residual of the boundary values by inserting the current values at the nodes"""
        # TODO: Make more efficient (save intermediate products)
        assert isinstance(self._z_gamma, jnp.ndarray)
        residual = np.einsum(
            'hi,hjk,j...k->i...', self._z_gamma, self._boundary_values,
            node_values) - self._z_gamma.T @ self._inhomogeneity
        return residual

    @property
    def n_inhom(self):
        return self._n_inhom

    @property
    def nnodes(self):
        return self._nnodes

    @property
    def boundary_values(self):
        return self._boundary_values

    @property
    def inhomogeneity(self):
        return self._inhomogeneity

    @inhomogeneity.setter
    def inhomogeneity(self, value):
        self._inhomogeneity = value

    @property
    def z_gamma(self):
        return self._z_gamma

    def set_z_gamma(self, rank: int, n: int) -> None:
        if self.z_gamma is None:
            z_gamma = np.zeros((n, rank), order='F')
            z_gamma[:rank, :rank] = np.identity(rank)
            self._z_gamma = z_gamma

    def _set_bvs(self, bvs: np.ndarray) -> np.ndarray:
        return np.array(list(bv.T for bv in bvs)).T


class BoundaryValues(MultipleBoundaryValues):
    """MultipleBoundaryValues subclass for the case of exactly two boundary values and :math:`q=2`."""
    def __init__(self,
                 boundary_0: np.ndarray,
                 boundary_f: np.ndarray,
                 inhomogeneity: np.ndarray,
                 z_gamma=None):
        """

        Parameters
        ----------
        boundary_0: boundary coefficient at :math:`t=t_0`
        boundary_f: boundary coefficient at :math:`t=t_f`
        inhomogeneity
        z_gamma
        """
        self.boundary_0: np.ndarray = boundary_0
        self.boundary_f: np.ndarray = boundary_f
        super().__init__((
            boundary_0,
            boundary_f,
        ), inhomogeneity, z_gamma)


class MultipleBoundaryValueProblem(VariableSolvable):
    """Baseclass for boundary value problems with multiple boundary values, possibly more than 2."""
    def __init__(self, time_intervals: Tuple[Time, ...], dynamical_system,
                 boundary_values: MultipleBoundaryValues):
        self._initial_time = time_intervals[0].t_0
        self._final_time = time_intervals[-1].t_f
        self._time_interval = Time(self._initial_time, self._final_time)
        self._time_intervals = time_intervals
        self._dynamical_system = dynamical_system
        self.nn: int = dynamical_system.nn
        self.nm: int = dynamical_system.nm
        self.boundary_values = boundary_values
        boundary_values.set_z_gamma(dynamical_system.rank, self.nn)
        if len(time_intervals) + 1 != self._boundary_values.nnodes:
            raise ValueError(
                "Timepoints and boundary values need to have same size.")
        self._nodes = self._get_and_check_nodes()
        variables = dynamical_system.variables
        super().__init__(variables)

    def solve(self, *args, **kwargs) -> 'pymloc.solvers.TimeSolution':
        retval = super().solve(*args, **kwargs)
        assert isinstance(retval, pymloc.solvers.TimeSolution)
        return retval

    @property
    def dynamical_system(self):
        """The corresponding dynamical system of the boundary value problem."""
        return self._dynamical_system

    @property
    def boundary_values(self) -> MultipleBoundaryValues:
        """The corresponding object storing the boundary condition"""
        return self._boundary_values

    @boundary_values.setter
    def boundary_values(self, value) -> None:
        self._boundary_values = value

    def _get_and_check_nodes(self):
        nodes = ()
        tis = self._time_intervals
        nodes += (tis[0].t_0, )
        for first_i, second_i in zip(tis, tis[1:]):
            tf = first_i.t_f
            t0 = second_i.t_0
            if not np.allclose(t0, tf):
                raise ValueError("TimeIntervals must intersect at boundary")
            nodes += (tf, )
        nodes += (tis[-1].t_f, )
        return nodes

    @property
    def nodes(self):
        return self._nodes

    @property
    def initial_time(self) -> float:
        """initial time of the boundary value problem"""
        return self._initial_time

    @property
    def final_time(self) -> float:
        """final time of the boundary value problem"""
        return self._final_time

    @property
    def time_interval(self) -> Time:
        """Time interval object."""
        return self._time_interval


class BoundaryValueProblem(MultipleBoundaryValueProblem):
    """Subclass for the case of exactly 2 boundary values."""
    def __init__(self, time_interval, dynamical_system,
                 boundary_values: BoundaryValues):
        super().__init__((time_interval, ), dynamical_system, boundary_values)

    @property
    def boundary_values(self) -> BoundaryValues:
        """The corresponding object storing the boundary condition"""
        return self._boundary_values

    @boundary_values.setter
    def boundary_values(self, value) -> None:
        self._boundary_values = value

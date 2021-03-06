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
from __future__ import annotations

from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np

import pymloc

from ...types import ParameterCallable
from ..multilevel_object import MultiLevelObject
from ..multilevel_object import local_object_factory
from ..variables import Time
from ..variables.container import VariablesContainer
from .boundary_value_problem import MultipleBoundaryValueProblem
from .boundary_value_problem import MultipleBoundaryValues
from .parameter_dae import jac_jax_reshaped


class ParameterMultipleBoundaryValues(MultiLevelObject):
    """Parameter dependent version of :class:`MultipleBoundaryValues`"""
    def __init__(self,
                 lower_level_variables: VariablesContainer,
                 higher_level_variables: VariablesContainer,
                 local_level_variables: VariablesContainer,
                 boundary_values: Sequence[ParameterCallable],
                 inhomogeneity: ParameterCallable,
                 nn: int,
                 n_param: int = 1,
                 z_gamma: ParameterCallable = None):
        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables)
        self._nnodes = len(boundary_values)
        self._boundary_values = boundary_values
        self._inhomogeneity = inhomogeneity
        self._z_gamma = z_gamma
        self._n_param = n_param
        self._nn = nn
        self._inhomogeneity_shape = (self._nn, )

    @property
    def boundary_values(self) -> Sequence[ParameterCallable]:
        return self._boundary_values

    @property
    def inhomogeneity(self) -> ParameterCallable:
        return self._inhomogeneity

    @inhomogeneity.setter
    def inhomogeneity(self, value):
        self._inhomogeneity = value

    def get_inhomogeneity_theta(self, solution: pymloc.solvers.TimeSolution,
                                parameters: np.ndarray) -> np.ndarray:
        """Computes the inhomogeneous part of the boundary conditions after
        differentiation with respect to parameters.

        Corresponds to the term :math:`\\tilde\\Gamma` in thesis."""
        # Assumes exactly 2-bounds. TODO: fix
        assert len(self.boundary_values) == 2
        sol = solution.solution
        inhom_theta = jac_jax_reshaped(self.inhomogeneity,
                                       self._inhomogeneity_shape)(parameters)
        shape = (self._nn, self._nn)
        boundary_indices = (0, -1)
        for bound, index in zip(self.boundary_values, boundary_indices):
            bound_theta = np.einsum('ijk,j->ik',
                                    jac_jax_reshaped(bound, shape)(parameters),
                                    sol[..., index])
            inhom_theta -= bound_theta
        return inhom_theta

    @property
    def z_gamma(self) -> Optional[ParameterCallable]:
        return self._z_gamma


class ParameterMultipleBoundaryValueProblem(MultiLevelObject):
    """Parameter dependent version of :class:`MultipleBoundaryValueProblem`"""
    _local_object_class = MultipleBoundaryValueProblem

    def __init__(
            self,
            lower_level_variables: VariablesContainer,
            higher_level_variables: VariablesContainer,
            local_level_variables: VariablesContainer,
            time_intervals: Tuple[Time, ...],
            dynamical_system,
            boundary_values: ParameterMultipleBoundaryValues,
    ):
        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables)

        self._time_intervals = time_intervals
        self._dynamical_system = dynamical_system
        self.nn = dynamical_system.nn
        self._boundary_values = boundary_values

    @property
    def dynamical_system(self):
        return self._dynamical_system

    @property
    def time_intervals(self) -> Tuple[Time, ...]:
        return self._time_intervals

    @property
    def boundary_values(self) -> ParameterMultipleBoundaryValues:
        return self._boundary_values


class ParameterBoundaryValueProblem(ParameterMultipleBoundaryValueProblem):
    def __init__(
            self,
            lower_level_variables: VariablesContainer,
            higher_level_variables: VariablesContainer,
            local_level_variables: VariablesContainer,
            time_interval: Time,
            dynamical_system,
            boundary_values: ParameterBoundaryValues,
    ):
        self._time_interval = time_interval
        self._initial_time = time_interval.t_0
        self._final_time = time_interval.t_f

        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables, (time_interval, ),
                         dynamical_system, boundary_values)

    @property
    def time_interval(self):
        return self._time_interval


class ParameterBoundaryValues(ParameterMultipleBoundaryValues):
    def __init__(self,
                 lower_level_variables: VariablesContainer,
                 higher_level_variables: VariablesContainer,
                 local_level_variables: VariablesContainer,
                 boundary_0: ParameterCallable,
                 boundary_f: ParameterCallable,
                 inhomogeneity: ParameterCallable,
                 nn: int,
                 n_param: int = 1,
                 z_gamma: Optional[ParameterCallable] = None):
        self.boundary_0 = boundary_0
        self.boundary_f = boundary_f
        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables, (
                             boundary_0,
                             boundary_f,
                         ), inhomogeneity, nn, n_param, z_gamma)


class AutomaticMultipleBoundaryValues(MultipleBoundaryValues):
    _auto_generated = True

    def __init__(self, global_object: ParameterMultipleBoundaryValues, *args,
                 **kwargs):
        self._global_object = global_object
        boundary_values = tuple(
            global_object.localize_method(method)
            for method in global_object.boundary_values)
        inhomogeneity = global_object.localize_method(
            global_object.inhomogeneity)
        z_gamma = global_object.localize_method(global_object.z_gamma)
        super().__init__(boundary_values, inhomogeneity, z_gamma=z_gamma)


local_object_factory.register_localizer(ParameterMultipleBoundaryValues,
                                        AutomaticMultipleBoundaryValues)

local_object_factory.register_localizer(ParameterBoundaryValues,
                                        AutomaticMultipleBoundaryValues)


class AutomaticMultipleBoundaryValueProblem(MultipleBoundaryValueProblem):
    _auto_generated = True

    def __init__(self, global_object: ParameterMultipleBoundaryValueProblem,
                 *args, **kwargs):
        parameters = kwargs.get('parameters')
        if parameters is not None:
            global_object.higher_level_variables.current_values = parameters
        self._global_object = global_object
        self._localization_parameters = parameters
        timepoints = global_object.time_intervals
        dynamical_system = global_object._dynamical_system.get_localized_object(
        )
        boundary_values = global_object.boundary_values.get_localized_object()

        super().__init__(timepoints, dynamical_system, boundary_values)


local_object_factory.register_localizer(ParameterMultipleBoundaryValueProblem,
                                        AutomaticMultipleBoundaryValueProblem)

local_object_factory.register_localizer(ParameterBoundaryValueProblem,
                                        AutomaticMultipleBoundaryValueProblem)

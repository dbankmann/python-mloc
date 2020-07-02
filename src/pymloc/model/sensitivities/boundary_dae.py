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

from ...types import ParameterCallable
from ..dynamical_system.boundary_value_problem import BoundaryValueProblem
from ..dynamical_system.parameter_bvp import ParameterBoundaryValueProblem
from ..dynamical_system.parameter_dae import LinearParameterDAE
from ..dynamical_system.parameter_dae import jac_jax_reshaped
from ..solvable import Solvable
from ..variables import Time


class BVPSensitivities(Solvable):
    """Solvable object that contains parameter dependent boundary value problems for strangenesss-free
    differential algebraic equations and all necessary quantities for the computation of the sensitivities.
    """
    def __init__(self,
                 boundary_value_problem: ParameterBoundaryValueProblem,
                 n_param: int,
                 selector: Optional[ParameterCallable] = None,
                 selector_shape: Optional[Tuple[int, ...]] = None):
        super().__init__()
        if not isinstance(boundary_value_problem,
                          ParameterBoundaryValueProblem):
            raise TypeError(
                "Only ParameterBoundaryValueProblem is supported at the moment"
            )
        self._bvp = boundary_value_problem
        self._dynamical_system = boundary_value_problem.dynamical_system
        self._time_interval = self._bvp.time_interval
        self._parameters = self._bvp.higher_level_variables
        self._n_param = n_param
        self._nn: int = self._dynamical_system.nn
        sel_shape: Tuple[int, ...]
        if selector is None:

            def selector(p: jnp.ndarray) -> jnp.ndarray:
                return np.identity(self._nn)

            sel_shape = (self._nn, self._nn)
        else:
            assert selector_shape is not None
            sel_shape = selector_shape
        self._selector_shape = sel_shape
        assert selector is not None
        self._selector: ParameterCallable = selector

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @property
    def n_param(self) -> int:
        return self._n_param

    @property
    def boundary_value_problem(self) -> ParameterBoundaryValueProblem:
        return self._bvp

    @property
    def dynamical_system(self) -> LinearParameterDAE:
        return self._dynamical_system

    @property
    def selector(self) -> ParameterCallable:
        return self._selector

    @property
    def selector_theta(self) -> ParameterCallable:
        shape = self._selector_shape
        return jac_jax_reshaped(self.selector, shape)

    @property
    def time_interval(self) -> Time:
        return self._time_interval

    def get_sensitivity_bvp(self, parameters=None) -> BoundaryValueProblem:
        return self._bvp.get_localized_object(parameters=parameters)

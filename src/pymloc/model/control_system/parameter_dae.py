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
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ...types import ParameterTimeCallable
from ..dynamical_system.parameter_dae import LinearParameterDAE
from ..multilevel_object import MultiLevelObject
from ..multilevel_object import local_object_factory
from ..variables import InputOutputStateVariables
from ..variables import Time
from ..variables.container import StateVariablesContainer
from .dae import LinearControlSystem


class LinearParameterControlSystem(MultiLevelObject):
    r"""Class for parameter dependent linear differential algebraic control systems of the form

.. math::
    E(t, \theta)\dot{x} &= A(t, \theta)x + B(t, \theta)u + f(t, \theta)\\
    y (t, \theta) &= C(t, \theta)x + D(t, \theta)u

    or (ommiting time and parameter arguments)

.. math::
    E(\frac{\mathrm d}{\mathrm dt}E^+E{x}) &= Ax +Bu + f\\
    y  &= Cx + Du.


    All coefficients are assumed sufficiently smooth.
    The system is assumed to be strangeness-free.
    All quantities according to the definitions in Kunkel, Mehrmann (2006) for every fixed parameter value."""

    _local_object_class = LinearControlSystem

    def __init__(self, ll_vars, hl_vars, loc_vars: InputOutputStateVariables,
                 e: ParameterTimeCallable, a: ParameterTimeCallable,
                 b: ParameterTimeCallable, c: ParameterTimeCallable,
                 d: ParameterTimeCallable, f: ParameterTimeCallable):
        super().__init__(ll_vars, hl_vars, loc_vars)
        self._e = e
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self._f = f
        if not isinstance(loc_vars, InputOutputStateVariables):
            raise TypeError(loc_vars)
        self.states = loc_vars.states
        self.inputs = loc_vars.inputs
        self.outputs = loc_vars.outputs
        self._nn = loc_vars.n_states
        self._nm = loc_vars.m_inputs
        self._np = loc_vars.p_outputs
        self._time = loc_vars.time
        cal_e, cal_a, cal_f = self._get_cal_coeffs(e, a, b, c, d, f)
        dim = self._nn + self._nm
        self._augmented_dae = LinearParameterDAE(ll_vars, hl_vars,
                                                 StateVariablesContainer(dim),
                                                 cal_e, cal_a, cal_f, dim)

        def _hom_f(p, t):
            return np.zeros((self._nn))

        self._free_dae = LinearParameterDAE(ll_vars, hl_vars,
                                            StateVariablesContainer(self._nn),
                                            e, a, _hom_f, self._nn)

    @property
    def local_level_variables(self) -> InputOutputStateVariables:
        return self._local_level_variables

    @local_level_variables.setter
    def local_level_variables(self, value):
        self._local_level_variables = value

    @property
    def e(self) -> ParameterTimeCallable:
        return self._e

    @property
    def a(self) -> ParameterTimeCallable:
        return self._a

    @property
    def b(self) -> ParameterTimeCallable:
        return self._b

    @property
    def c(self) -> ParameterTimeCallable:
        return self._c

    @property
    def d(self) -> ParameterTimeCallable:
        return self._d

    @property
    def f(self) -> ParameterTimeCallable:
        return self._f

    @property
    def time(self) -> Time:
        return self._time

    @property
    def augmented_dae(self) -> LinearParameterDAE:
        return self._augmented_dae

    @property
    def free_dae(self) -> LinearParameterDAE:
        return self._free_dae

    # TODO: Refactor. Identical to control dae
    def _get_cal_coeffs(self, e: ParameterTimeCallable,
                        a: ParameterTimeCallable, b: ParameterTimeCallable,
                        c: ParameterTimeCallable, d: ParameterTimeCallable,
                        f: ParameterTimeCallable
                        ) -> Tuple[ParameterTimeCallable, ...]:
        nn = self._nn
        nm = self._nm

        @jax.jit
        def cal_e(*args, **kwargs):
            cal_e_arr = jnp.block([[e(*args, **kwargs), jnp.zeros((nn, nm))]])
            return cal_e_arr

        @jax.jit
        def cal_a(*args, **kwargs):
            cal_a_arr = jnp.block([[a(*args, **kwargs), b(*args, **kwargs)]])
            return cal_a_arr

        @jax.jit
        def cal_f(*args, **kwargs):
            cal_f_arr = f(*args, **kwargs)
            return cal_f_arr

        return cal_e, cal_a, cal_f


class AutomaticLinearControlSystem(LinearControlSystem):
    def __init__(self, parameter_dae: LinearParameterControlSystem):
        self._parameter_dae = parameter_dae
        variables: InputOutputStateVariables = parameter_dae.local_level_variables
        e, a, b, c, d, f = (parameter_dae.localize_method(method)
                            for method in (parameter_dae.e, parameter_dae.a,
                                           parameter_dae.b, parameter_dae.c,
                                           parameter_dae.d, parameter_dae.f))

        super().__init__(variables, e, a, b, c, d, f)


local_object_factory.register_localizer(LinearParameterControlSystem,
                                        AutomaticLinearControlSystem)

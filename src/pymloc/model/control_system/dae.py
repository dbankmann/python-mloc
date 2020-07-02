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

import jax.numpy as jnp
import numpy as np

from ...types import TimeCallable
from .. import Solvable
from ..dynamical_system.dae import LinearDAE
from ..variables import StateVariablesContainer
from ..variables import Time
from ..variables.container import InputOutputStateVariables


class LinearControlSystem(Solvable):
    r"""Class for linear differential algebraic control systems of the form

    .. math::
        E\dot{x} &= Ax + Bu + f\\
        y &= C x + Du

    or

    .. math::
        E\frac{\mathrm d}{\mathrm dt}(E^+E{x}) &= Ax + f\\
        y &= C x + Du.

    All coefficients are assumed sufficiently smooth. The system is assumed to be strangeness-free.
    All quantities according to the definitions in Kunkel, Mehrmann (2006)."""
    def __init__(self, variables: InputOutputStateVariables, e: TimeCallable,
                 a: TimeCallable, b: TimeCallable, c: TimeCallable,
                 d: TimeCallable, f: TimeCallable):
        self._e = e
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        if not isinstance(variables, InputOutputStateVariables):
            raise TypeError(variables)
        self.states = variables.states
        self.inputs = variables.inputs
        self.outputs = variables.outputs
        self._nn = variables.n_states
        self._nm = variables.m_inputs
        self._np = variables.p_outputs
        self._time = variables.time
        cal_e, cal_a, cal_f = self._get_cal_coeffs(
            e, a, b, c, d,
            f)  # type: (TimeCallable, TimeCallable, TimeCallable)
        dim = self._nn + self._nm
        self._augmented_dae = LinearDAE(StateVariablesContainer(dim), cal_e,
                                        cal_a, cal_f, dim)

        def _hom_f(t: float) -> np.ndarray:
            return np.zeros((self._nn))

        self._free_dae = LinearDAE(StateVariablesContainer(self._nn), e, a,
                                   _hom_f, self._nn)

    def reset(self) -> None:
        """Resets all DAE objects. Removes stored current values."""
        self._free_dae.reset()
        self._augmented_dae.reset()

    def _get_cal_coeffs(self, e: TimeCallable, a: TimeCallable,
                        b: TimeCallable, c: TimeCallable, d: TimeCallable,
                        f: TimeCallable) -> Tuple[TimeCallable, ...]:
        nn = self._nn
        nm = self._nm

        def cal_e(*args, **kwargs):
            cal_e_arr = jnp.block([[e(*args, **kwargs), jnp.zeros((nn, nm))]])
            return cal_e_arr

        def cal_a(*args, **kwargs):
            cal_a_arr = jnp.block([[a(*args, **kwargs), b(*args, **kwargs)]])
            return cal_a_arr

        def cal_f(*args, **kwargs):
            cal_f_arr = f(*args, **kwargs)
            return cal_f_arr

        return cal_e, cal_a, cal_f

    @property
    def time(self) -> Time:
        return self._time

    @property
    def nm(self) -> int:
        """Number of input variables."""
        return self._nm

    @property
    def nn(self) -> int:
        """Number of states."""
        return self._nn

    @property
    def np(self) -> int:
        """Number of output variables."""
        return self._np

    @property
    def e(self) -> TimeCallable:
        return self._e

    @property
    def a(self) -> TimeCallable:
        return self._a

    @property
    def b(self) -> TimeCallable:
        return self._b

    @property
    def c(self) -> TimeCallable:
        return self._c

    @property
    def d(self) -> TimeCallable:
        return self._d

    @property
    def augmented_dae(self) -> LinearDAE:
        """Corresponding linear DAE by treating both state and input as variables."""
        return self._augmented_dae

    @property
    def free_dae(self) -> LinearDAE:
        """Corresponding DAE where input is neglected and removed from the equation, i.e., set to 0."""
        return self._free_dae

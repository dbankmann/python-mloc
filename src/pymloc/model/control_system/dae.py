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
import jax.numpy as jnp
import numpy as np

from ..dynamical_system.dae import LinearDAE
from ..variables import InputOutputStateVariables
from ..variables.container import StateVariablesContainer


class LinearControlSystem:
    def __init__(self, variables, e, a, b, c, d, f):
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
        cal_e, cal_a, cal_f = self._get_cal_coeffs(e, a, b, c, d, f)
        dim = self._nn + self._nm
        self._augmented_dae = LinearDAE(StateVariablesContainer(dim), cal_e,
                                        cal_a, cal_f, dim)

        def _hom_f(t):
            return np.zeros((self._nn))

        self._free_dae = LinearDAE(StateVariablesContainer(self._nn), e, a,
                                   _hom_f, self._nn)

    def reset(self):
        self._free_dae.reset()
        self._augmented_dae.reset()

    def _get_cal_coeffs(self, e, a, b, c, d, f):
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
    def time(self):
        return self._time

    @property
    def nm(self):
        return self._nm

    @property
    def nn(self):
        return self._nn

    @property
    def np(self):
        return self._np

    @property
    def e(self):
        return self._e

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    @property
    def d(self):
        return self._d

    @property
    def augmented_dae(self):
        return self._augmented_dae

    @property
    def free_dae(self):
        return self._free_dae

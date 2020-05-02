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
import logging

import jax.numpy as jnp
import numpy as np

from ...model.variables.container import StateVariablesContainer

logger = logging.getLogger(__name__)


class DAE:
    def __init__(self, variables, n):
        if not isinstance(variables, StateVariablesContainer):
            raise TypeError(variables)
        self._variables = variables
        self._nm = self._variables.n_states
        if n != self._nm and n != self._nm[0]:
            raise ValueError(
                "Number of variables is {}, but has to equal number of equations, which is {}"
                .format(self._nm, n))
        self._nn = n
        self._index = None
        self._current_t = None
        self._current_rank = None

    @property
    def nm(self):
        return self._nm

    @property
    def nn(self):
        return self._nn

    @property
    def index(self):
        return self._index

    @property
    def variables(self):
        return self._variables


class LinearDAE(DAE):
    def __init__(self,
                 variables,
                 e,
                 a,
                 f,
                 n,
                 der_e=None,
                 constant_coefficients=True):
        self._constant_coefficients = constant_coefficients
        super().__init__(variables, n)
        self._e = e
        self._a = a
        self._f = f
        self._rank = None
        self.reset()
        # TODO: Make more general!
        self.init_rank()
        if der_e is None:
            der_e = self._der_e_numerical
        self._der_e = der_e
        self._current_ahat = np.zeros((n, n), order='F')
        self._current_ehat = np.zeros((n, n), order='F')
        var_shape = self._variables.n_states
        self._current_fhat = np.zeros(var_shape, order='F')

    def reset(self):
        logger.debug("Current a:\n{}\n{}".format(self._a, self._a(0.)))
        self._current_t = dict()

    @property
    def constant_coefficients(self):
        return self._constant_coefficients

    @constant_coefficients.setter
    def constant_coefficients(self, value):
        self._constant_coefficients = value

    @property
    def rank(self):
        if self._rank is None:
            raise ValueError("Rank has to be initialized first.")
        return self._rank

    def init_rank(self):
        # TODO: Choose meaningful timepoint
        self._compute_rank(0.)

    def _compute_rank(self, t):
        e = self.e(t)
        rank = jnp.linalg.matrix_rank(e)
        if self._current_rank is not None and rank != self._current_rank:
            raise ValueError(
                "Rank change in parameters detected. Not supported and may lead to wrong results."
            )
        self._rank = rank

    def der_e(self, t):
        return self._der_e(t)

    def _der_e_numerical(self, t):
        # Use tools like jax
        h = 10e-5
        e_h = self.e(t + h)
        e = self.e(t)
        return (e_h - e) / h

    def _check_current_time(self, t, method, time_varying=False):
        if self._current_t.get(method) is None or (
            (time_varying or not self.constant_coefficients)
                and self._current_t[method] != t):
            self._current_t[method] = t
            return True
        else:
            return False

    def e(self, t):
        self._recompute_coefficients(t)
        return self._current_e

    def a(self, t):
        self._recompute_coefficients(t)
        return self._current_a

    def f(self, t):
        self._recompute_inhomogeinity(t)
        return self._current_f

    def eplus(self, t):
        self._recompute_quantities(t)
        return self._current_eplus

    def t2(self, t):
        self._recompute_quantities(t)
        return self._current_ttprime_h[:, :self.rank]

    def t2prime(self, t):
        self._recompute_quantities(t)
        return self._current_ttprime_h[:, self.rank:]

    def z1(self, t):
        self._recompute_quantities(t)
        return self._current_zzprime[:, :self.rank]

    def z1prime(self, t):
        self._recompute_quantities(t)
        return self._current_zzprime[:, self.rank:]

    def ehat_1(self, t):
        self._recompute_quantities(t)
        return self._current_ehat[:self.rank, :]

    def ehat_2(self, t):
        self._recompute_quantities(t)
        return self._current_ehat[self.rank:, :]

    def ahat_1(self, t):
        self._recompute_quantities(t)
        return self._current_ahat[:self.rank, :]

    def ahat_2(self, t):
        self._recompute_quantities(t)
        return self._current_ahat[self.rank:, :]

    def fhat_1(self, t):
        self._recompute_fhat(t)
        return self._current_fhat[:self.rank, ...]

    def fhat_2(self, t):
        self._recompute_fhat(t)
        return self._current_fhat[self.rank:, ...]

    def _recompute_coefficients(self, t):
        if self._check_current_time(t, "coefficients"):
            e = self._e(t)
            a = self._a(t)
            self._current_e = e
            self._current_a = a

    def _recompute_inhomogeinity(self, t):
        if self._check_current_time(t, "inhomogeinity", time_varying=True):
            f = self._f(t)
            self._current_f = f

    def _recompute_quantities(self, t):
        if self._check_current_time(t, "quantities"):
            e = self.e(t)
            a = self.a(t)
            zzprime, sigma, ttprime_h = jnp.linalg.svd(e)
            rank = self.rank
            self._current_ttprime_h = ttprime_h.T
            self._current_zzprime = zzprime
            self._current_eplus = self.t2(t) @ np.linalg.solve(
                np.diag(sigma[:rank]),
                self.z1(t).T)
            ehat_1 = self.z1(t).T @ e
            self._current_ehat[:rank, :] = ehat_1
            ahat_1 = self.z1(t).T @ a
            ahat_2 = self.z1prime(t).T @ a
            self._current_ahat[:rank, :] = ahat_1
            self._current_ahat[rank:, :] = ahat_2

    def _recompute_fhat(self, t):
        self._recompute_quantities(t)
        if self._check_current_time(t, "fhat", time_varying=True):
            rank = self.rank
            f = self.f(t)
            fhat_1 = self.z1(t).T @ f
            fhat_2 = self.z1prime(t).T @ f
            self._current_fhat[:rank, ...] = fhat_1
            self._current_fhat[rank:, ...] = fhat_2

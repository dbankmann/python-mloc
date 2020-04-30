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
import numpy as np
import scipy.integrate
from scipy.integrate import trapz

from . import LocalObjective


class LQRObjective(LocalObjective):
    def __init__(self, time, q, s, r, final_weight):
        self._time = time
        self._q = q
        self._s = s
        self._r = r
        self._m = final_weight
        self.reset()
        super().__init__()

    def reset(self):
        self._current_t = dict()

    def integral_weights(self, t):
        self._recompute_int_weights(t)
        return self._current_integral_weights

    def _recompute_int_weights(self, t):
        if self._check_current_time(t, 'int_weights'):
            self._current_q = self._q(t)
            self._current_r = self._r(t)
            self._current_s = self._s(t)
            q = self._current_q
            r = self._current_r
            s = self._current_s
            weights = np.block([[q, s], [s.T.conj(), r]])
            self._current_integral_weights = weights

    #TODO: Refactor dae and objective to time objects
    def _check_current_time(self, t, method):
        if self._current_t.get(method) is None or self._current_t[method] != t:
            self._current_t[method] = t
            return True
        else:
            return False

    def value(self, solution):
        t0 = self._time.t_0
        tf = self._time.t_f
        xtf = solution(tf)
        final = xtf.T @ self.final_weights @ xtf
        solmatrix = solution.solution
        intweights = np.array(
            [self.integral_weights(t) for t in self._time.grid])
        integral = trapz(
            np.einsum('ti,tij,tj->t', solution, intweights, solution), t0, tf)

        return integral + intweights

    @property
    def final_weight(self):
        return self._m

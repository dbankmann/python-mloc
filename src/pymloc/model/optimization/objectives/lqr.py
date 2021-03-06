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
from typing import Dict

import numpy as np
from scipy.integrate import trapz

import pymloc

from ....types import TimeCallable
from ...variables import Time
from . import LocalObjective


class LQRObjective(LocalObjective):
    r"""Implementation of the objective function of the linear quadratic regulator.

    The objective function is given by

    .. math::
        {
        x(\tf)^{\Herm}Kx(\tf) + \int_{t_0}^{\tf}{
            \begin{pmatrix}
                x(t)\\
                u(t)
            \end{pmatrix}^{\Herm}
            \begin{bmatrix}
                Q(t)  & S(t)\\
                S^{\Herm}(t)& R(t)
            \end{bmatrix}
            \begin{pmatrix}
                x(t)\\
                u(t)
            \end{pmatrix}
    \mathrm{d}t}}.
    """
    def __init__(self, time: Time, q: TimeCallable, s: TimeCallable,
                 r: TimeCallable, final_weight: np.ndarray):
        """
        Parameter
        ---------

        time: time interval of integration with :math:`t_0` as starting point and :math:`t_f` as end point.
        q: weight function
        r: weight function
        s: weight function
        final_weight: final weight
        """
        self._time = time
        self._q = q
        self._s = s
        self._r = r
        self._m = final_weight
        self._current_t: Dict[str, float] = dict()
        self.reset()
        super().__init__()

    def reset(self):
        """Resets all DAE objects. Removes stored current values."""
        self._current_t = dict()

    def integral_weights(self, t: float) -> np.ndarray:
        self._recompute_int_weights(t)
        return self._current_integral_weights

    def _recompute_int_weights(self, t: float) -> np.ndarray:
        if self._check_current_time(t, 'int_weights'):
            self._current_q = self._q(t)
            self._current_r = self._r(t)
            self._current_s = self._s(t)
            q = self._current_q
            r = self._current_r
            s = self._current_s
            weights = np.block([[q, s], [s.T.conj(), r]])
            self._current_integral_weights = weights

    # TODO: Refactor dae and objective to time objects
    def _check_current_time(self, t: float, method: str) -> bool:
        if self._current_t.get(method) is None or self._current_t[method] != t:
            self._current_t[method] = t
            return True
        else:
            return False

    def value(self, solution: "pymloc.solvers.TimeSolution") -> np.ndarray:
        t0 = self._time.t_0
        tf = self._time.t_f
        xtf = solution(tf)
        final = xtf.T @ self.final_weight @ xtf
        solmatrix = solution.solution
        intweights = np.array(
            [self.integral_weights(t) for t in self._time.grid])
        integral = trapz(
            np.einsum('ti,tij,tj->t', solmatrix, intweights, solmatrix), t0,
            tf)

        return integral + final

    @property
    def final_weight(self) -> np.ndarray:
        return self._m

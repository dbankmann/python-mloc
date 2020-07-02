#
# Copyright (c) 2019-2020
#
# @autho: floatr->np.ndarray: Daniel Bankmann
# @company: Technische Universität Berlin
#
# This file is part of the python package pymloc
# (see https://gitlab.tubit.tu-berlin.de/bankmann91/python-mloc )
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#

import numpy as np

from .dae import LinearDAE


class LinearFlowRepresentation(LinearDAE):
    """Adds additional methods for the computation of quantities of the flow formulation for strangeness-free DAEs.

    See Baum, 2015 for more details."""
    def __init__(self, variables, e, a, f, n, der_e=None):
        super().__init__(variables, e, a, f, n, der_e)

    def projection(self, t: float) -> np.ndarray:
        self._compute_projection(t)
        return self._current_projection

    def projection_complement(self, t: float) -> np.ndarray:
        self._compute_projection(t)
        return self._current_proj_compl

    def _compute_projection(self, t: float) -> np.ndarray:
        if not self._check_current_time(t, "projection"):
            return
        self._recompute_quantities(t)
        self._current_projection = self.t2(t) @ self.t2(t).T
        self._current_proj_compl = -self._current_projection + np.identity(
            self.nn)

    def cal_projection(self, t: float) -> np.ndarray:
        self._compute_cal_proj(t)
        return self._current_cal_proj

    def _compute_cal_proj(self, t: float) -> np.ndarray:
        self._compute_projection(t)
        self._current_cal_proj = (np.identity(self.nn) -
                                  self.d_a(t)) @ self.projection(t)

    def d_d(self, t: float) -> np.ndarray:
        self._compute_d_d(t)
        return self._current_d_d

    def _compute_d_d(self, t: float) -> np.ndarray:
        # TODO: save intermediate
        epa = self._compute_eplusa(t)
        self._current_d_d = epa @ self.cal_projection(t)

    def _compute_eplusa(self, t: float) -> np.ndarray:
        epa = (self.t2(t) @ np.linalg.solve(
            self.ehat_1(t) @ self.t2(t), self.ahat_1(t)) +
               self.projection_derivative(t))
        return epa

    def d_a(self, t: float) -> np.ndarray:
        self._compute_d_a(t)
        return self._current_d_a

    def _compute_d_a(self, t: float) -> np.ndarray:
        temp = self.ahat_2(t) @ self.t2prime(t)
        self._current_d_a = self.t2prime(t) @ np.linalg.solve(
            temp, self.ahat_2(t))

    def f_a(self, t: float) -> np.ndarray:
        self._compute_f_a(t)
        return self._current_f_a

    def _compute_f_a(self, t: float) -> np.ndarray:
        temp = self.ahat_2(t) @ self.t2prime(t)
        self._current_f_a = self.t2prime(t) @ np.linalg.solve(
            temp, self.fhat_2(t))

    def f_d(self, t: float) -> np.ndarray:
        self._compute_f_d(t)
        return self._current_f_d

    def x_d(self, t, x: float) -> np.ndarray:
        self._compute_projection(t)
        return self._current_projection @ x

    def x_a(self, t, x: float) -> np.ndarray:
        self._compute_projection(t)
        return self._current_proj_compl @ x

    def _compute_f_d(self, t: float) -> np.ndarray:
        self._compute_projection_derivative(t)
        epa = self._compute_eplusa(t)
        self._current_f_d = self.t2(t) @ np.linalg.solve(
            self.ehat_1(t) @ self.t2(t), self.fhat_1(t)) - epa @ self.f_a(t)

    def projection_derivative(self, t: float) -> np.ndarray:
        self._compute_projection_derivative(t)
        return self._current_proj_derivative

    def _compute_projection_derivative(self, t: float) -> np.ndarray:
        if self._check_current_time(t, "proj_derivative"):
            eplus = self.eplus(t)
            e = self.e(t)
            der_e = self.der_e(t)
            n = self.nn
            der_ep_e = -eplus @ der_e @ eplus @ e + (
                np.identity(n) - eplus @ e) @ der_e.T @ eplus.T @ eplus @ e
            ep_der_e = eplus @ der_e
            self._current_proj_derivative = der_ep_e + ep_der_e

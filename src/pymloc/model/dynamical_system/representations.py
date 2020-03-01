import numpy as np

from .dae import LinearDAE


class LinearFlowRepresentation(LinearDAE):
    def __init__(self, variables, e, a, f, n, der_e=None):
        super().__init__(variables, e, a, f, n, der_e)

    def projection(self, t):
        self._compute_projection(t)
        return self._current_projection

    def projection_complement(self, t):
        self._compute_projection(t)
        return self._current_proj_compl

    def _compute_projection(self, t):
        if self._check_current_time(t, "projection"):
            return
        self._recompute_quantities(t)
        self._current_projection = self.t2(t) @ self.t2(t).T
        self._current_proj_compl = -self._current_projection + np.identity(
            self.nn)

    def cal_projection(self, t):
        self._compute_cal_proj(t)
        return self._current_cal_proj

    def _compute_cal_proj(self, t):
        self._compute_projection(t)
        self._current_cal_proj = (np.identity(self.nn) -
                                  self.d_a(t)) @ self.projection(t)

    def d_d(self, t):
        self._compute_d_d(t)
        return self._current_d_d

    def _compute_d_d(self, t):
        self._current_d_d = (
            self.t2(t) @ np.linalg.solve(
                self.ehat_1(t) @ self.t2(t), self.ahat_1(t)) +
            self.projection_derivative(t)) @ self.cal_projection(t)

    def d_a(self, t):
        self._compute_d_a(t)
        return self._current_d_a

    def _compute_d_a(self, t):
        temp = self.ahat_2(t) @ self.t2prime(t)
        self._current_d_a = self.t2prime(t) @ np.linalg.solve(
            temp, self.ahat_2(t))

    def f_a(self, t):
        self._compute_f_a(t)
        return self._current_f_a

    def _compute_f_a(self, t):
        temp = self.ahat_2(t) @ self.t2prime(t)
        self._current_f_a = self.t2prime(t) @ np.linalg.solve(
            temp, self.fhat_2(t))

    def f_d(self, t):
        self._compute_f_d(t)
        return self._current_f_d

    def _compute_f_d(self, t):
        self._current_f_d = (self.t2(t) @ np.linalg.solve(
            self.ehat_1(t) @ self.t2(t), self.fhat_1(t)) +
                             self.proj_derivative(t)) @ self.cal_projection(t)

    def projection_derivative(self, t):
        self._compute_projection_derivative(t)
        return self._current_proj_derivative

    def _compute_projection_derivative(self, t):
        eplus = self._current_eplus
        e = self.e(t)
        der_e = self.der_e(t)
        n = self.nn
        der_ep_e = -eplus @ der_e @ eplus @ e + (
            np.identity(n) - eplus @ e) @ der_e.T @ eplus.T @ eplus @ e
        ep_der_e = eplus @ der_e
        self._current_proj_derivative = der_ep_e + ep_der_e

import numpy as np

from ..dynamical_system.parameter_dae import LinearParameterDAE
from ..multilevel_object import MultiLevelObject


class LinearParameterControlSystem(LinearParameterDAE):
    def __init__(self, ll_vars, hl_vars, loc_vars, e, a, b, c, d, f, nn):
        self._e = e
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        cal_e, cal_a, cal_f = self._get_cal_coeffs(e, a, b, c, d, f)
        super().__init__(ll_vars, hl_vars, loc_vars, cal_e, cal_a, cal_f, nn)

    def _get_cal_coeffs(self, e, a, b, c, d, f):
        def cal_e(ll_vars, loc_vars):
            znm = np.zeros()
            np.linalg.block([
                e,
            ])

        def cal_a(ll_vars, loc_vars):
            znm = np.zeros()
            np.linalg.block([
                e,
            ])

        def cal_f(ll_vars, loc_vars):
            znm = np.zeros()
            np.linalg.block([
                e,
            ])

        return cal_e, cal_a, cal_f

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

    def residual(self, p, x, xdot, u, t):
        e = self.e(p, x, t)
        a = self.a(p, x, t)
        b = self.b(p, x, t)
        c = self.c(p, x, t)
        d = self.d(p, x, t)
        return e @ xdot - a @ x - b @ u

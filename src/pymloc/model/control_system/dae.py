from ..multilevel_object import MultiLevelObject


class ParameterControlSystem(MultiLevelObject):
    def __init__(self,
                 lower_level_variables,
                 higher_level_variables,
                 local_level_variables,
                 residual=None):
        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables)
        if residual is not None:
            self.residual = residual


class LinearParameterControlSystem(ParameterControlSystem):
    def __init__(self, ll_vars, hl_vars, loc_vars, e, a, b, c, d):
        super().__init__(ll_vars, hl_vars, loc_vars)
        self._e = e
        self._a = a
        self._b = b
        self._c = c
        self._d = d

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

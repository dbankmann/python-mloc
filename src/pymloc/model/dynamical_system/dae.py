import numpy as np

from ...model.variables.time_function import StateVariables
from ..multilevel_object import MultiLevelObject
from ..multilevel_object import local_object_factory


class ParameterDAE(MultiLevelObject):
    def __init__(self,
                 lower_level_variables,
                 higher_level_variables,
                 local_level_variables,
                 residual=None):
        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables)
        if residual is not None:
            self.residual = residual


class LinearParameterDAE(ParameterDAE):
    def __init__(self, ll_vars, hl_vars, loc_vars, e, a, f):
        super().__init__(ll_vars, hl_vars, loc_vars)
        self._e = e
        self._a = a
        self._f = f

    @property
    def e(self):
        return self._e

    @property
    def a(self):
        return self._a

    def residual(self, hl_vars, loc_vars, ll_vars):
        p, = hl_vars.current_value
        xdot, x, t = loc_vars.current_value
        e = self.e(p, x, t)
        a = self.a(p, x, t)
        return e @ xdot - a @ x


class DAE:
    def __init__(self, variables, n):
        if not isinstance(variables, StateVariables):
            raise TypeError(variables)
        self._variables = variables
        self._nm = variables.dimension
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


class LinearDAE(DAE):
    def __init__(self, variables, e, a, f, n):
        super().__init__(variables, n)
        self._e = e
        self._a = a
        self._f = f
        self._rank = None

    @property
    def e(self):
        return self._e

    @property
    def a(self):
        return self._a

    @property
    def f(self):
        return self._f

    @property
    def rank(self):
        return self._rank

    def _compute_projection(self, t):
        if self._current_t is None or self._current_t != t:
            self._current_t = t
            e = self.e(t)
            n = self.nn
            zzprime, sigma, ttprime_h = np.linalg.svd(e)
            rank = np.linalg.matrix_rank(e)
            if self._current_rank is not None and rank != self._current_rank:
                raise ValueError(
                    "Rank change in parameters detected. Not supported and may lead to wrong results."
                )
            self._rank = rank
            self._current_t2 = ttprime_h[:, :rank]
            self._current_t2prime = ttprime_h[:, rank:]
            self._current_z1 = zzprime[:, :rank]
            self._current_z1prime = zzprime[:, rank:]

    def get_t2(self, t):
        self._compute_projection(t)
        return self._current_t2


class AutomaticLinearDAE(LinearDAE):
    def __init__(self, parameter_dae, *args, **kwargs):
        self._parameter_dae = parameter_dae
        variables = parameter_dae.local_level_variables
        super().__init__(variables, *args, **kwargs)


local_object_factory.register_localizer(LinearParameterDAE, AutomaticLinearDAE)

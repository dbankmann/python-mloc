import numpy as np

from ...model.variables.time_function import StateVariables
from ..multilevel_object import MultiLevelObject, local_object_factory


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
    def __init__(self, variables, e, a, f, n, der_e=None):
        super().__init__(variables, n)
        self._e = e
        self._a = a
        self._f = f
        self._rank = None
        self._current_t = dict()
        #TODO: Make more general!
        self.init_rank()
        if der_e is None:
            der_e = lambda t: self._der_e_numerical(t)
        self._der_e = der_e

    @property
    def rank(self):
        if self._rank is None:
            raise ValueError("Rank has to be initialized first.")
        return self._rank

    def init_rank(self):
        #TODO: Choose meaningful timepoint
        self._compute_rank(0.)

    def _compute_rank(self, t):
        e = self.e(t)
        rank = np.linalg.matrix_rank(e)
        if self._current_rank is not None and rank != self._current_rank:
            raise ValueError(
                "Rank change in parameters detected. Not supported and may lead to wrong results."
            )
        self._rank = rank

    def der_e(self, t):
        return self._der_e(t)

    def _der_e_numerical(self, t):
        #Use tools like jax
        n = self._nn
        h = 10e-5
        e_h = self.e(t + h)
        e = self.e(t)
        return (e_h - e) / h

    def _check_current_time(self, t, method):
        if self._current_t.get(method) is None or self._current_t[method] != t:
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
        self._recompute_coefficients(t)
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
        self._recompute_quantities(t)
        return self._current_fhat[:self.rank, :]

    def fhat_2(self, t):
        self._recompute_quantities(t)
        return self._current_fhat[self.rank:, :]

    def _recompute_coefficients(self, t):
        if self._check_current_time(t, "coefficients"):
            e = self._e(t)
            a = self._a(t)
            f = self._f(t)
            self._current_e = e
            self._current_a = a
            self._current_f = f

    def _recompute_quantities(self, t):
        if self._check_current_time(t, "quantities"):
            e = self.e(t)
            a = self.a(t)
            f = self.f(t)
            n = self.nn
            zzprime, sigma, ttprime_h = np.linalg.svd(e)
            rank = self.rank
            self._current_ttprime_h = ttprime_h
            self._current_zzprime = zzprime
            self._current_eplus = self.t2(t) @ np.linalg.solve(
                np.diag(sigma[:rank]),
                self.z1(t).T)
            ehat_1 = self.z1(t).T @ e
            self._current_ehat = np.zeros((n, n), order='F')
            self._current_ehat[:rank, :] = ehat_1
            ahat_1 = self.z1(t).T @ a
            ahat_2 = self.z1prime(t).T @ a
            self._current_ahat = np.zeros((n, n), order='F')
            self._current_ahat[:rank, :] = ahat_1
            self._current_ahat[rank:, :] = ahat_2
            fhat_1 = self.z1(t).T @ f
            fhat_2 = self.z1prime(t).T @ f
            self._current_fhat = np.zeros((n, ))
            self._current_fhat[:rank] = fhat_1
            self._current_fhat[rank:] = fhat_2


class AutomaticLinearDAE(LinearDAE):
    def __init__(self, parameter_dae, *args, **kwargs):
        self._parameter_dae = parameter_dae
        variables = parameter_dae.local_level_variables
        super().__init__(variables, *args, **kwargs)


local_object_factory.register_localizer(LinearParameterDAE, AutomaticLinearDAE)

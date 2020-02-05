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
    def __init__(self, variables):
        self._variables = variables
        self._index = None

    @property
    def index(self):
        return self._index


class LinearDAE(DAE):
    def __init__(self, variables, e, a, f):
        super().__init__(variables)
        self._cal_e = e
        self._cal_a = a
        self._cal_f = f


class AutomaticLinearDAE(LinearDAE):
    def __init__(self, parameter_dae, *args, **kwargs):
        self._parameter_dae = parameter_dae
        variables = parameter_dae.local_level_variables
        super().__init__(variables, *args, **kwargs)


local_object_factory.register_localizer(LinearParameterDAE, AutomaticLinearDAE)

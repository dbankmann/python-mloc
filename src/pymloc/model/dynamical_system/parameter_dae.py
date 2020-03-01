from ..multilevel_object import MultiLevelObject
from ..multilevel_object import local_object_factory
from .representations import LinearFlowRepresentation


class ParameterDAE(MultiLevelObject):
    def __init__(self,
                 lower_level_variables,
                 higher_level_variables,
                 local_level_variables,
                 n,
                 residual=None):
        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables)
        if residual is not None:
            self.residual = residual
        self.nn = n


class LinearParameterDAE(ParameterDAE):
    def __init__(self, ll_vars, hl_vars, loc_vars, e, a, f, n, der_e=None):
        self._e = e
        self._a = a
        self._f = f
        self._der_e = der_e
        super().__init__(ll_vars, hl_vars, loc_vars, n, self.residual)

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
    def der_e(self):
        return self._der_e

    def residual(self, hl_vars, loc_vars, ll_vars):
        p, = hl_vars.current_value
        xdot, x, t = loc_vars.current_value
        e = self.e(p, x, t)
        a = self.a(p, x, t)
        return e @ xdot - a @ x


#TODO: Automatic generation should take into consideration appropriate representations. For now LinearFlowRepresentation is general enough.
class AutomaticLinearDAE(LinearFlowRepresentation):
    def __init__(self, parameter_dae):
        self._parameter_dae = parameter_dae
        variables = parameter_dae.local_level_variables
        nn = parameter_dae.nn
        e, a, f, der_e = (parameter_dae.localize_method(method)
                          for method in (parameter_dae.e, parameter_dae.a,
                                         parameter_dae.f, parameter_dae.der_e))
        super().__init__(variables, e, a, f, nn, der_e)


local_object_factory.register_localizer(LinearParameterDAE, AutomaticLinearDAE)

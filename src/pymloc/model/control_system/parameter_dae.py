import numpy as np

from ..dynamical_system.parameter_dae import LinearParameterDAE
from ..multilevel_object import MultiLevelObject
from ..multilevel_object import local_object_factory
from ..variables import InputOutputStateVariables
from .dae import LinearControlSystem


class LinearParameterControlSystem(MultiLevelObject):
    _local_object_class = LinearControlSystem

    def __init__(self, ll_vars, hl_vars, loc_vars, e, a, b, c, d, f):
        super().__init__(ll_vars, hl_vars, loc_vars)
        self._e = e
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self._f = f
        if not isinstance(loc_vars, InputOutputStateVariables):
            raise TypeError(loc_vars)
        self.states = loc_vars.states
        self.inputs = loc_vars.inputs
        self.outputs = loc_vars.outputs
        self._nn = loc_vars.n_states
        self._nm = loc_vars.m_inputs
        self._np = loc_vars.p_outputs

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

    @property
    def f(self):
        return self._f


class AutomaticLinearControlSystem(LinearControlSystem):
    def __init__(self, parameter_dae):
        self._parameter_dae = parameter_dae
        variables = parameter_dae.local_level_variables
        e, a, b, c, d, f = (parameter_dae.localize_method(method)
                            for method in (parameter_dae.e, parameter_dae.a,
                                           parameter_dae.b, parameter_dae.c,
                                           parameter_dae.d, parameter_dae.f))
        super().__init__(variables, e, a, b, c, d, f)


local_object_factory.register_localizer(LinearParameterControlSystem,
                                        AutomaticLinearControlSystem)

import jax
import jax.numpy as jnp
import numpy as np

from ..dynamical_system.parameter_dae import LinearParameterDAE
from ..multilevel_object import MultiLevelObject
from ..multilevel_object import local_object_factory
from ..variables import InputOutputStateVariables
from ..variables.container import StateVariablesContainer
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
        self._time = loc_vars.time
        cal_e, cal_a, cal_f = self._get_cal_coeffs(e, a, b, c, d, f)
        dim = self._nn + self._nm
        self._augmented_dae = LinearParameterDAE(ll_vars, hl_vars,
                                                 StateVariablesContainer(dim),
                                                 cal_e, cal_a, cal_f, dim)

        _hom_f = lambda p, t: np.zeros((self._nn))
        self._free_dae = LinearParameterDAE(ll_vars, hl_vars,
                                            StateVariablesContainer(self._nn),
                                            e, a, _hom_f, self._nn)

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

    @property
    def time(self):
        return self._time

    @property
    def augmented_dae(self):
        return self._augmented_dae

    @property
    def free_dae(self):
        return self._free_dae

    #TODO: Refactor. Identical to control dae
    def _get_cal_coeffs(self, e, a, b, c, d, f):
        nn = self._nn
        nm = self._nm

        @jax.jit
        def cal_e(*args, **kwargs):
            cal_e_arr = jnp.block([[e(*args, **kwargs), jnp.zeros((nn, nm))]])
            return cal_e_arr

        @jax.jit
        def cal_a(*args, **kwargs):
            cal_a_arr = jnp.block([[a(*args, **kwargs), b(*args, **kwargs)]])
            return cal_a_arr

        @jax.jit
        def cal_f(*args, **kwargs):
            cal_f_arr = f(*args, **kwargs)
            return cal_f_arr

        return cal_e, cal_a, cal_f


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

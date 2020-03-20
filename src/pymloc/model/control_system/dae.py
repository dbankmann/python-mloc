import numpy as np

from ..dynamical_system.dae import LinearDAE
from ..multilevel_object import MultiLevelObject
from ..variables import InputOutputStateVariables
from ..variables.container import StateVariablesContainer


class LinearControlSystem:
    def __init__(self, variables, e, a, b, c, d, f):
        self._e = e
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        if not isinstance(variables, InputOutputStateVariables):
            raise TypeError(variables)
        self.states = variables.states
        self.inputs = variables.inputs
        self.outputs = variables.outputs
        self._nn = variables.n_states
        self._nm = variables.m_inputs
        self._np = variables.p_outputs
        self._time = variables.time
        cal_e, cal_a, cal_f = self._get_cal_coeffs(e, a, b, c, d, f)
        dim = self._nn + self._nm
        self._augmented_dae = LinearDAE(StateVariablesContainer(dim), cal_e,
                                        cal_a, cal_f, dim)

    def _get_cal_coeffs(self, e, a, b, c, d, f):
        shape = (self._nn, self._nn + self._nm)
        cal_e_arr = np.zeros(shape)
        cal_a_arr = np.zeros(shape)
        cal_f_arr = np.zeros(self._nn)

        def cal_e(*args, **kwargs):
            cal_e_arr[:self._nn, :self._nn] = e(*args, **kwargs)
            return cal_e_arr

        def cal_a(*args, **kwargs):
            cal_a_arr[:self._nn, :self._nn] = a(*args, **kwargs)
            cal_a_arr[:self._nn, self._nn:] = b(*args, **kwargs)
            return cal_a_arr

        def cal_f(*args, **kwargs):
            cal_f_arr[:] = f(*args, **kwargs)
            return cal_f_arr

        return cal_e, cal_a, cal_f

    @property
    def time(self):
        return self._time

    @property
    def nm(self):
        return self._nm

    @property
    def nn(self):
        return self._nn

    @property
    def np(self):
        return self._np

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
    def augmented_dae(self):
        return self._augmented_dae

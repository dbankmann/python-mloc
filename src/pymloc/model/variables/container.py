from abc import ABC

import numpy as np

from .discrete import Parameters
from .time_function import InputVariables
from .time_function import OutputVariables
from .time_function import StateVariables
from .time_function import Time
from .variables import Variables


class VariablesContainer(ABC):
    def __init__(self):
        self._variables = []

    @property
    def variables(self):
        return self._variables

    def merge(self, *args):
        for variables in args:
            if isinstance(variables, Variables):
                self._variables.append(variables)
            else:
                raise TypeError(
                    "{} must be a subclass of Variables".format(variables))

    @property
    def current_values(self):
        vals = [var.current_values for var in self.variables]
        return vals

    def get_random_values(self):
        vals = (var.get_random_values() for var in self.variables)
        return vals


class UniqueVariablesContainer(VariablesContainer, ABC):
    @property
    def current_values(self):
        vals = self.variables[0].current_values
        return vals

    @current_values.setter
    def current_values(self, value):
        var = self.variables[0]
        var.current_values = value

    def get_random_values(self):
        vals = (var.get_random_values() for var in self.variables)
        return vals


class InputOutputStateVariables(VariablesContainer):
    def __init__(self, n_states, m_inputs, p_outputs, time=Time(0., 1.)):
        super().__init__()
        self._n_states = n_states
        self._m_inputs = m_inputs
        self._p_outputs = p_outputs
        #TODO: Check consistency of times
        self._states = StateVariables(n_states)
        self._inputs = InputVariables(m_inputs)
        self._outputs = OutputVariables(p_outputs)
        self._time = time
        self._merge()

    def _check_time_domains(self, *args):
        pass

    def _merge(self):
        self.merge(self.states, self.inputs, self.outputs, self.time)

    @property
    def n_states(self):
        return self._n_states

    @property
    def m_inputs(self):
        return self._m_inputs

    @property
    def p_outputs(self):
        return self._p_outputs

    @property
    def states(self):
        return self._states

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def time(self):
        return self._time


class InputStateVariables(InputOutputStateVariables):
    def __init__(self, n_states, m_inputs, **kwargs):
        super().__init__(n_states, m_inputs, 0, **kwargs)
        self._outputs = None

    def _merge(self):
        self.merge(self.states, self.inputs, self.time)


class StateVariablesContainer(InputStateVariables):
    def __init__(self, n_states, **kwargs):
        super().__init__(n_states, 0, **kwargs)
        self._inputs = None

    def _merge(self):
        self.merge(self.states, self.inputs, self.time)

    @property
    def shape(self):
        return np.array(self.dimension).shape


class ParameterContainer(UniqueVariablesContainer):
    def __init__(self, p_parameters, domain):
        super().__init__()
        self._parameters = Parameters(p_parameters, domain)
        self.merge(self.parameters)

    @property
    def parameters(self):
        return self._parameters

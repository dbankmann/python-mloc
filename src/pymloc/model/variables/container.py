from .variables import Variables
from .time_function import StateVariables, InputVariables, OutputVariables
from abc import ABC


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


class InputOutputStateVariables(VariablesContainer):
    def __init__(self, n_states, m_inputs, p_outputs):
        super().__init__()
        self._n_states = n_states
        self._m_inputs = m_inputs
        self._p_outputs = p_outputs
        self._states = StateVariables(n_states)
        self._inputs = InputVariables(m_inputs)
        self._outputs = OutputVariables(p_outputs)
        self.merge(self.states, self.inputs, self.outputs)

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

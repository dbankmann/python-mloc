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
        self._n_variables = 0
        self._linked_variable = None

    @property
    def variables(self):
        return self._variables

    @property
    def linked_variable(self):
        return self._linked_variable

    @linked_variable.setter
    def linked_variable(self, value):
        self._linked_variable = value

    def link_variable(self, variable, overwrite=False):
        self.linked_variable = variable
        if overwrite:
            variable.time = self._time

    def merge(self, *args):
        for variables in args:
            if isinstance(variables, Variables):
                self._variables.append(variables)
                self._n_variables += 1
            else:
                raise TypeError(
                    "{} must be a subclass of Variables".format(variables))

    @property
    def current_values(self):
        #TODO: Refactor
        if self._linked_variable is None:
            return self._local_values()
        else:
            return self._linked_variable.current_values

    def _local_values(self):
        if self._n_variables == 1:
            return self.variables[0].current_values
        else:
            vals = [var.current_values for var in self.variables]
            return vals

    @current_values.setter
    def current_values(self, values):
        if self._n_variables == 1:
            self.variables[0].current_values = values
        else:
            for value, variable in zip(values, self.variables):
                variable.current_values = value

    def get_random_values(self):
        vals = (var.get_random_values() for var in self.variables)
        return vals

    def update_values(self):
        try:
            problem = self.associated_problem
        except:
            raise ValueError("There is no associated problem or solver")

        if problem is not None:
            problem.init_solver()
            solution = problem.solve()

    def get_sensitivities(self):
        sens_obj = self.sensitivity_problem.get_sensitivities()
        sens_obj.init_solver()
        sens = sens_obj.solve()
        return sens

    @property
    def associated_problem(self):
        return self._associated_problem

    @associated_problem.setter
    def associated_problem(self, value):
        self._associated_problem = value

    @property
    def sensitivity_problem(self):
        return self._sensitivity_problem

    @sensitivity_problem.setter
    def sensitivity_problem(self, value):
        self._sensitivity_problem = value


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
        if self._linked_variable is None:
            return self._time
        else:
            return self._linked_variable.time

    @time.setter
    def time(self, value):
        self._time = value


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
        self.merge(self.states)

    @property
    def shape(self):
        n = self._n_states
        if isinstance(n, tuple):
            return n
        else:
            return (n, )


class ParameterContainer(UniqueVariablesContainer):
    def __init__(self, p_parameters, domain):
        super().__init__()
        self._parameters = Parameters(p_parameters, domain)
        self.merge(self.parameters)

    @property
    def parameters(self):
        return self._parameters

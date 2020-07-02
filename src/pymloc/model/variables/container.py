#
# Copyright (c) 2019-2020
#
# @author: Daniel Bankmann
# @company: Technische Universität Berlin
#
# This file is part of the python package pymloc
# (see https://gitlab.tubit.tu-berlin.de/bankmann91/python-mloc )
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#
from __future__ import annotations

import logging
from abc import ABC
from typing import List
from typing import Optional

import numpy as np

import pymloc

from .discrete import Parameters
from .time_function import InputVariables
from .time_function import OutputVariables
from .time_function import StateVariables
from .time_function import Time
from .variables import Variables

logger = logging.getLogger(__name__)


class VariablesContainer(ABC):
    """Base class for all variable containers."""
    def __init__(self):
        self._variables: List = []
        self._n_variables: int = 0
        self._linked_variable: Optional[VariablesContainer] = None

    @property
    def variables(self) -> List[Variables]:
        return self._variables

    @property
    def linked_variable(self) -> Optional[VariablesContainer]:
        return self._linked_variable

    @linked_variable.setter
    def linked_variable(self, value):
        self._linked_variable = value

    def merge(self, *args: Variables) -> None:
        for variables in args:
            if isinstance(variables, Variables):
                self._variables.append(variables)
                self._n_variables += 1
            else:
                raise TypeError(
                    "{} must be a subclass of Variables".format(variables))

    @property
    def current_values(self) -> np.ndarray:
        # TODO: Refactor
        if self._linked_variable is None:
            return self._local_values()
        else:
            return self._linked_variable.current_values

    @current_values.setter
    def current_values(self, values):
        if self._n_variables == 1:
            self.variables[0].current_values = values
        else:
            for value, variable in zip(values, self.variables):
                variable.current_values = value

    def _local_values(self) -> np.ndarray:
        if self._n_variables == 1:
            return self.variables[0].current_values
        else:
            vals = [var.current_values for var in self.variables]
            return vals

    def get_random_values(self) -> np.ndarrray:
        vals = (var.get_random_values() for var in self.variables)
        return vals

    def _check_and_init(self, problem: 'pymloc.model.Solvable',
                        **kwargs) -> None:
        if len(kwargs) == 0:
            return
        if not problem.has_solver_instance():
            logger.warning(
                "Solver not initialized for problem: {}.\nInitializing with defaults..."
                .format(problem))
        problem.init_solver(**kwargs)

    def update_values(self, **kwargs) -> Optional['pymloc.solvers.Solution']:
        try:
            problem = self.associated_problem
        except AttributeError:
            raise AttributeError("There is no associated problem or solver")

        if problem is not None:
            self._check_and_init(problem, **kwargs)
            solution = problem.solve()
            return solution

        return None

    def get_sensitivities(self, **kwargs) -> np.ndarray:
        sens_obj = self.sensitivity_problem
        self._check_and_init(sens_obj, **kwargs)
        sens = sens_obj.solve()
        return sens

    @property
    def associated_problem(self) -> 'pymloc.model.Solvable':
        return self._associated_problem

    @associated_problem.setter
    def associated_problem(self, value):
        self._associated_problem = value

    @property
    def sensitivity_problem(self) -> 'pymloc.model.Solvable':
        return self._sensitivity_problem

    @sensitivity_problem.setter
    def sensitivity_problem(self, value):
        self._sensitivity_problem = value


class UniqueVariablesContainer(VariablesContainer, ABC):
    """Subclass, with only one variable object in the container"""
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
    """VariablesContainer with states, inputs and outputs."""
    def __init__(self,
                 n_states: int,
                 m_inputs: int,
                 p_outputs: int,
                 time: Time = Time(0., 1.)):
        super().__init__()
        self._n_states = n_states
        self._m_inputs = m_inputs
        self._p_outputs = p_outputs
        # TODO: Check consistency of times
        self._states = StateVariables(n_states)
        self._inputs = InputVariables(m_inputs)
        self._outputs = OutputVariables(p_outputs)
        self._time = time
        self._merge()

    def _check_time_domains(self, *args):
        pass

    def link_variable(self,
                      variable: InputOutputStateVariables,
                      overwrite: bool = False) -> None:
        self.linked_variable = variable
        if overwrite:
            variable.time = self._time

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
    """VariablesContainer with states and inputs."""
    def __init__(self, n_states, m_inputs, **kwargs):
        super().__init__(n_states, m_inputs, 0, **kwargs)
        self._outputs = None

    def _merge(self):
        self.merge(self.states, self.inputs, self.time)


class StateVariablesContainer(InputStateVariables):
    """VariablesContainer with states."""
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
    """"VariablesContainer for parameters"""
    def __init__(self, p_parameters: int,
                 domain: 'pymloc.model.domains.RNDomain'):
        super().__init__()
        self._parameters = Parameters(p_parameters, domain)
        self.merge(self.parameters)

    @property
    def parameters(self) -> Parameters:
        return self._parameters

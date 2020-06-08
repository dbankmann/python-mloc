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
import logging
from abc import ABC

from ..model.variables.container import VariablesContainer
from ..solver_container import solver_container_factory

logger = logging.getLogger()


class Solvable(ABC):
    _auto_generated: bool = False

    def __init__(self):
        self.__get_class()
        self._available_solvers = solver_container_factory.get_solver_container(
            self)
        self.set_default_solver()
        self._solver_instance = None

    def set_default_solver(self):
        if self._available_solvers is None:
            logger.error("No solvers available. Cannot set default solver.")
        else:
            self._solver = self._available_solvers.default_solver

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        self._solver = solver

    @property
    def available_solver(self):
        return self._available_solvers

    def _save_solution(self, solution):
        pass

    def solve(self, *args, **kwargs):
        try:
            solution = self._solver_instance.run(*args, **kwargs)
            self._save_solution(solution)
            return solution
        except AttributeError:
            raise AttributeError(
                "There is currently no instantiated solver object for problem {}"
                .format(self))

    def init_solver(self, *args, **kwargs):
        creator_func = self._solver.creator_function
        if creator_func is None:
            tmp = self
        else:
            tmp = creator_func(self)
        if not isinstance(tmp, tuple):
            tmp = tuple((tmp, ))
        self._solver_instance = self._solver.solver(*tmp, *args, **kwargs)

    @property
    def solver_instance(self):
        return self._solver_instance

    def has_solver_instance(self):
        return self._solver_instance is not None

    def __get_class(self):
        if self._auto_generated:
            self._class = self._global_object._local_object_class
        else:
            self._class = self.__class__


class VariableSolvable(Solvable, ABC):
    def __init__(self, variables: VariablesContainer):
        super().__init__()
        if not isinstance(variables, VariablesContainer):
            raise TypeError(variables)
        self._variables = variables
        self._associate_problem_variables()

    def _associate_problem_variables(self):
        self.variables.associated_problem = self

    @property
    def variables(self):
        return self._variables

    def _save_solution(self, solution):
        self.variables.set_value = solution

    def link_solution(self, variables):
        pass

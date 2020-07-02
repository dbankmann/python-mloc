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
from typing import Optional

import pymloc

from ..model.variables.container import VariablesContainer
from ..solver_container import SolverContainer
from ..solver_container import SolverTuple
from ..solver_container import solver_container_factory

logger = logging.getLogger()


class Solvable(ABC):
    """Abstract base class for solvable problems.
    Needs at least one solver registered via solver_container_factory"""
    _auto_generated: bool = False
    _global_object: Optional['pymloc.model.MultiLevelObject']

    def __init__(self):
        self.__get_class()
        self._available_solvers: SolverContainer = solver_container_factory.get_solver_container(
            self)
        self.set_default_solver()
        self._solver_instance: Optional['pymloc.solvers.BaseSolver'] = None

    def set_default_solver(self) -> None:
        if self._available_solvers is None:
            logger.error("No solvers available. Cannot set default solver.")
        else:
            self.solver = self._available_solvers.default_solver

    @property
    def solver(self) -> SolverTuple:
        return self._solver

    @solver.setter
    def solver(self, solver):
        self._solver = solver

    @property
    def available_solver(self) -> SolverContainer:
        return self._available_solvers

    def _save_solution(self, solution: 'pymloc.solvers.Solution') -> None:
        ...

    def solve(self, *args, **kwargs) -> 'pymloc.solvers.Solution':
        try:
            assert self._solver_instance is not None
            solution = self._solver_instance.run(*args, **kwargs)
            self._save_solution(solution)
            return solution
        except AssertionError:
            raise AttributeError(
                "There is currently no instantiated solver object for problem {}"
                .format(self))

    def init_solver(self, *args, **kwargs) -> None:
        """Initializes solver object.

        The solver is initialized via solver factory and the corresponding creator_function.

        If the creator function does not exist, the solver can be used directly on the Solvable
        object.
        """
        creator_func = self._solver.creator_function
        if creator_func is None:
            tmp1 = self
        else:
            tmp1 = creator_func(self)
        if not isinstance(tmp1, tuple):
            tmp = tuple((tmp1, ))
        else:
            tmp = tmp1
        self._solver_instance = self._solver.solver(*tmp, *args, **kwargs)

    @property
    def solver_instance(self) -> Optional['pymloc.solvers.BaseSolver']:
        return self._solver_instance

    def has_solver_instance(self) -> bool:
        return self._solver_instance is not None

    def __get_class(self) -> None:
        if self._auto_generated:
            try:
                assert self._global_object is not None
                self._class = self._global_object.local_object_class
            except AssertionError:
                raise AttributeError(
                    "Auto generated local solvables need a corresponding _global_object attribute"
                )
        else:
            self._class = self.__class__


class VariableSolvable(Solvable, ABC):
    """Subclass for Solvable that have corresponding variables."""
    def __init__(self, variables: VariablesContainer):
        super().__init__()
        if not isinstance(variables, VariablesContainer):
            raise TypeError(variables)
        self.variables = variables
        self._associate_problem_variables()

    def _associate_problem_variables(self) -> None:
        self.variables.associated_problem = self

    @property
    def variables(self) -> VariablesContainer:
        return self._variables

    @variables.setter
    def variables(self, value):
        self._variables = value

    def _save_solution(self, solution):
        self.variables.set_value = solution

    def link_solution(self, variables: VariablesContainer) -> None:
        pass

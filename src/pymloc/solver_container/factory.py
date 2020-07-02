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

import inspect
import logging
from typing import Callable
from typing import ClassVar
from typing import Optional
from typing import Type

import pymloc

from ..model import solvable
from . import SolverContainer

logger = logging.getLogger()


class SolverContainerFactory:
    """Factory class for solver containers.

    Maintains the mapping between models (Solvables) and Containers of solvers."""
    __instance: ClassVar[Optional[SolverContainerFactory]] = None

    @staticmethod
    def get_instance():
        if SolverContainerFactory.__instance is None:
            SolverContainerFactory()
        return SolverContainerFactory.__instance

    def __init__(self):
        if SolverContainerFactory.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            SolverContainerFactory.__instance = self
        self._solvers = dict()

    def _check_problem(self, problem: Type[solvable.Solvable]):
        if not inspect.isclass(problem):
            raise TypeError(problem)

    def _check_problem_instance(self, problem_instance: solvable.Solvable):
        if not isinstance(problem_instance, solvable.Solvable):
            raise TypeError(problem_instance)

    def _get_problem_class(self, problem_instance: solvable.Solvable):
        return problem_instance._class

    def register_solver(self,
                        problem: Type[solvable.Solvable],
                        solver: 'pymloc.solvers.BaseSolver',
                        default: bool = False,
                        creator_function: Optional[Callable] = None):
        """"Registers a solver for the given problem.

        Parameters
        ----------
        problem: The solvable
        solver: A solver compliant with the solvable
        default: Whether this solver should be *the* default solver for problem.
        creator_function: A method of Solvable, that can be used to create another solvable for
            which other types of solvers exist. This is most suitable for higher level abstraction.
            E.g., optimal control problems :class:`pymloc.model.optimization.LQOptimalControl` can
            be solved in multiple ways. One approach is via necessary conditions, which itself constitute
            a :class:`pymloc.model.dynamical_system.boundary_value_problem.BoundaryValueProblem`.
            Hence, solver :class:`pymloc.solvers.dynamical_systems.multiple_shooting.MultipleShooting`
            can bes used to solve :class:`pymloc.model.optimization.LQOptimalControl` with the
            creator_function :method:`pymloc.model.optimization.LQOptimalControl.get_bvp`.
        """
        self._check_problem(problem)
        solver_container = self._solvers.get(problem)
        if solver_container is None:
            self._solvers[problem] = SolverContainer(problem, solver, default,
                                                     creator_function)
        else:
            solver_container.add_solver(solver, default, creator_function)
        # TODO: Also register solvers for all problem subclasses.
        for subclass in problem.__subclasses__():
            self.register_solver(subclass, solver, False, creator_function)

    def get_solver_container(self, problem_instance: solvable.Solvable
                             ) -> SolverContainer:
        self._check_problem_instance(problem_instance)
        problem = self._get_problem_class(problem_instance)
        solver_container = self._solvers.get(problem)
        if solver_container is None:
            logger.error(
                "No registered solvers for problem: {}".format(problem))
        return solver_container


solver_container_factory = SolverContainerFactory.get_instance()

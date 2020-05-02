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
import inspect
import logging

from ..model import solvable
from . import SolverContainer

logger = logging.getLogger()


class SolverContainerFactory:
    __instance = None

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

    def _check_problem(self, problem):
        if not inspect.isclass(problem):
            raise TypeError(problem)

    def _check_problem_instance(self, problem_instance):
        if not isinstance(problem_instance, solvable.Solvable):
            raise TypeError(problem_instance)

    def _get_problem_class(self, problem_instance):
        return problem_instance._class

    def register_solver(self,
                        problem,
                        solver,
                        default=False,
                        creator_function=None):
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

    def get_solver_container(self, problem_instance):
        self._check_problem_instance(problem_instance)
        problem = self._get_problem_class(problem_instance)
        solver_container = self._solvers.get(problem)
        if solver_container is None:
            logger.error(
                "No registered solvers for problem: {}".format(problem))
        return solver_container


solver_container_factory = SolverContainerFactory.get_instance()

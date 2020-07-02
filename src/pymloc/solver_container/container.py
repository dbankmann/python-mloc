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
from typing import Callable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Type

import pymloc


class SolverTuple(NamedTuple):
    solver: 'pymloc.solvers.BaseSolver'
    creator_function: Optional[Callable]


class SolverContainer:
    """Container class that holds Solver objects."""
    def __init__(self, problem: Type['pymloc.model.Solvable'],
                 solver: 'pymloc.solvers.BaseSolver', default: bool,
                 creator_function: Optional[Callable]):
        self._problem = problem
        self._solvers: List[SolverTuple] = []
        self.add_solver(solver, default, creator_function)

    def add_solver(self,
                   solver: 'pymloc.solvers.BaseSolver',
                   default: bool = False,
                   creator_function: Optional[Callable] = None) -> None:
        solver_tuple = SolverTuple(solver, creator_function)
        self._solvers.append(solver_tuple)
        if default or not hasattr(self, "_default_solver"):
            self.default_solver = solver_tuple

    @property
    def problem(self) -> Type['pymloc.model.Solvable']:
        return self._problem

    @property
    def solvers(self) -> List[SolverTuple]:
        return self._solvers

    @property
    def default_solver(self) -> SolverTuple:
        return self._default_solver

    @default_solver.setter
    def default_solver(self, solver_tuple):
        if solver_tuple not in self.solvers:
            raise ValueError("{} object should be a registered solver".format(
                solver_tuple.solver))
        self._default_solver = solver_tuple

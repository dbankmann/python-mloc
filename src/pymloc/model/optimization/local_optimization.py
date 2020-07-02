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
from abc import ABC

from ..solvable import VariableSolvable
from ..variables.container import VariablesContainer
from .constraints.local_constraint import LocalConstraint
from .objectives.local_objective import LocalObjective


class LocalOptimizationObject(VariableSolvable, ABC):
    """
    Abstract class for defining local optimization problems.
    """
    def __init__(self, loc_objective_obj: LocalObjective,
                 loc_constraint_obj: LocalConstraint,
                 variables_obj: VariablesContainer):

        if not isinstance(loc_objective_obj, LocalObjective):
            raise TypeError(loc_objective_obj)
        if not isinstance(loc_constraint_obj, LocalConstraint):
            raise TypeError(loc_constraint_obj)
        self.objective = loc_objective_obj
        self.constraint = loc_constraint_obj
        super().__init__(variables_obj)

    @property
    def objective(self) -> LocalObjective:
        return self._objective

    @objective.setter
    def objective(self, value):
        self._objective = value

    @property
    def constraint(self) -> LocalConstraint:
        return self._constraint

    @constraint.setter
    def constraint(self, value):
        self._constraint = value


class LocalNullOptimization(LocalOptimizationObject):
    """Dummy implementation of a local optimization."""
    def residual(self):
        return 0.

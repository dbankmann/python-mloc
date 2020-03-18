from abc import ABC
from abc import abstractmethod

from ..solvable import Solvable
from ..variables.container import VariablesContainer
from .constraints.local_constraint import LocalConstraint
from .objectives.local_objective import LocalObjective


class LocalOptimizationObject(Solvable, ABC):
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
        if not isinstance(variables_obj, VariablesContainer):
            raise TypeError(variables_obj)
        self._objective = loc_objective_obj
        self._constraint = loc_constraint_obj
        self._variables = variables_obj
        super().__init__()

    @property
    def objective(self):
        return self._objective

    @property
    def constraint(self):
        return self._constraint

    @property
    def variables(self):
        return self._variables


class LocalNullOptimization(LocalOptimizationObject):
    def residual(self):
        return 0.

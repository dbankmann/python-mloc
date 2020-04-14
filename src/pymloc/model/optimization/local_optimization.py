from abc import ABC
from abc import abstractmethod

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
        self._objective = loc_objective_obj
        self._constraint = loc_constraint_obj
        super().__init__(variables_obj)

    @property
    def objective(self):
        return self._objective

    @property
    def constraint(self):
        return self._constraint


class LocalNullOptimization(LocalOptimizationObject):
    def residual(self):
        return 0.

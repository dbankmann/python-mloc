from ..solvable import solver_container_factory
from ...solvers.solver import NullSolver
from ..solvable import Solvable
from .constraints.local_constraint import LocalConstraint
from .objectives.local_objective import LocalObjective
from ..variables.container import VariablesContainer
from abc import ABC, abstractmethod


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
        self._loc_objective_object = loc_objective_obj
        self._loc_constraint_object = loc_constraint_obj
        self._variables = variables_obj
        self.__get_class()
        super().__init__()

    @property
    def loc_objective_object(self):
        return self._loc_objective_object

    @property
    def loc_constraint_object(self):
        return self._loc_constraint_object

    @property
    def variables(self):
        return self._variables

    def __get_class(self):
        if self._auto_generated:
            self._class = self._global_optimization._local_object_class
        else:
            self._class = self.__class__


class LocalNullOptimization(LocalOptimizationObject):
    def residual(self):
        return 0.


solver_container_factory.register_solver(LocalNullOptimization,
                                         NullSolver,
                                         default=True)

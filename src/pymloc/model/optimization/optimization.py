from ..multilevel_object import MultiLevelObject, local_object_factory
from .constraints.constraint import Constraint
from .objectives.objective import Objective
from ..variables.container import VariablesContainer
from .local_optimization import LocalOptimizationObject, LocalNullOptimization
from abc import ABC


class OptimizationObject(MultiLevelObject, ABC):
    """
    Class for defining optimization problems.

    """
    def __init__(self, objective_obj: Objective, constraint_obj: Constraint,
                 lower_level_variables: VariablesContainer,
                 higher_level_variables: VariablesContainer,
                 local_level_variables: VariablesContainer):
        if not isinstance(objective_obj, Objective):
            raise TypeError(objective_obj)
        if not isinstance(constraint_obj, Constraint):
            raise TypeError(constraint_obj)
        for variable_container in higher_level_variables, lower_level_variables, local_level_variables:
            if not isinstance(variable_container, VariablesContainer):
                raise TypeError(variable_container)
        self._objective_object = objective_obj
        self._constraint_object = constraint_obj
        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables)

    @property
    def objective_object(self):
        return self._objective_object

    @property
    def constraint_object(self):
        return self._constraint_object


class NullOptimization(OptimizationObject):
    _local_object_class = LocalNullOptimization


class AutomaticLocalOptimizationObject(LocalOptimizationObject):
    _auto_generated = True

    def __init__(self, global_optimization, *args, **kwargs):
        self._global_object = global_optimization
        loc_objective = global_optimization.objective_object.get_localized_object(
        )
        loc_constraint = global_optimization.constraint_object.get_localized_object(
        )
        loc_vars = global_optimization.local_level_variables
        super().__init__(loc_objective, loc_constraint, loc_vars, *args,
                         **kwargs)


local_object_factory.register_localizer(NullOptimization,
                                        AutomaticLocalOptimizationObject)

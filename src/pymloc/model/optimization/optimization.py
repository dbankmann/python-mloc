from .multilevel_object import MultiLevelObject
from .constraints.constraint import Constraint
from .objectives.objective import Objective
from ..variables.container import VariablesContainer
from .local_optimization import LocalOptimizationObject


class OptimizationObject(MultiLevelObject):
    """
    Class for defining optimization problems.

    """
    def __init__(self, objective_obj: Objective, constraint_obj: Constraint,
                 lower_level_variables: VariablesContainer,
                 higher_level_variables: VariablesContainer,
                 local_level_variables: VariablesContainer):
        if not isinstance(objective_obj, Objective):
            raise TypeError
        if not isinstance(constraint_obj, Constraint):
            raise TypeError
        for variable_container in higher_level_variables, lower_level_variables, local_level_variables:
            if not isinstance(variable_container, VariablesContainer):
                raise TypeError
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

    def residual(self):
        raise NotImplementedError

    def get_localized_object(self):
        '''Method that generically initializes a LocalOptimizationObject.
        Should be overloaded for improved performance'''
        loc_objective = self.objective_object.get_localized_object()
        loc_constraint = self.constraint_object.get_localized_object()
        return LocalOptimizationObject(loc_objective, loc_constraint,
                                       self.local_level_variables)


class NullOptimization(OptimizationObject):
    def __init__(self):
        pass

    def get_localized_object(self):
        pass

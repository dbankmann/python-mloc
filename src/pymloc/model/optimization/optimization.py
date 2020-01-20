from .constraints.constraint import Constraint
from .objectives.objective import Objective
from ..variables.variables import Variables

class OptimizationObject(object):
    """
    Class for defining optimization problems.

    """
    def __init__(self, objective_obj: Objective, constraint_obj: Constraint, variables_obj: Variables):
        if not isinstance(objective_obj, Objective):
            raise TypeError
        if not isinstance(constraint_obj, Constraint):
            raise TypeError
        if not isinstance(variables_obj, Variables):
            raise TypeError
        self.objective_object = objective_obj
        self.constraint_object = constraint_obj


    def residual(self):
        raise NotImplementedError



class NullOptimization(OptimizationObject):
    def __init__(self):
        pass

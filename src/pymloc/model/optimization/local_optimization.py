from .constraints.local_constraint import LocalConstraint
from .objectives.local_objective import LocalObjective


class LocalOptimizationObject(object):
    """
    Class for defining local optimization problems.
    """

    def __init__(self, loc_objective_obj: LocalObjective, loc_constraint_obj: LocalConstraint):

        self.loc_objective_object = loc_objective_obj
        self.loc_constraint_object = loc_constraint_obj


    def residual(self):
        raise NotImplementedError



class LocalNullOptimization(LocalOptimizationObject):
    def __init__(self):
        pass

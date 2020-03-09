from ..multilevel_object import local_object_factory
from ..solvable import Solvable
from .constraints.constraint import NullConstraint
from .optimization import AutomaticLocalOptimizationObject
from .optimization import OptimizationObject


class LocalNonLinearLeastSquares(Solvable):
    pass


class NonLinearLeastSquares(OptimizationObject):
    _local_object_class = LocalNonLinearLeastSquares

    def __init__(self, objective_obj, ll_vars, hl_vars, loc_vars):
        vars = (ll_vars, hl_vars, loc_vars)
        super().__init__(objective_obj, NullConstraint(*vars), *vars)


class AutomaticLocalNonLinearLeastSquares(AutomaticLocalOptimizationObject):
    pass


local_object_factory.register_localizer(NonLinearLeastSquares,
                                        AutomaticLocalNonLinearLeastSquares)

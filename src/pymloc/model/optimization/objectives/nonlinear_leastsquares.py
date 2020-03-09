from ...multilevel_object import MultiLevelObject
from ...multilevel_object import local_object_factory
from .objective import AutomaticLocalObjective
from .objective import Objective


class NonLinearLeastSquares(Objective):
    def __init__(self, lower_level_variables, higher_level_variables,
                 local_level_variables, rhs):
        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables)
        self._rhs = rhs

    def residual(self, ll_vars, hl_vars, loc_vars):
        return self._rhs(ll_vars, hl_vars, loc_vars)


class AutomaticLocalNonLinearLeastSquares(AutomaticLocalObjective):
    pass


local_object_factory.register_localizer(NonLinearLeastSquares,
                                        AutomaticLocalNonLinearLeastSquares)

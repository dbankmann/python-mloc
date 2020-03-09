from ...multilevel_object import local_object_factory
from ...variables import NullVariables
from . import AutomaticLocalObjective
from . import Objective


class ParameterLQRObjective(Objective):
    def __init__(self, higher_level_variables, local_level_variables,
                 integral_weights, final_weights):
        lower_level_variables = NullVariables()
        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables)
        self._integral_weigts = integral_weights
        self._final_weights = final_weights


class AutomaticLocalLQRObjective(AutomaticLocalObjective):
    pass


local_object_factory.register_localizer(ParameterLQRObjective,
                                        AutomaticLocalLQRObjective)

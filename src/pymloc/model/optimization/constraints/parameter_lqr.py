from ...dynamical_system.parameter_dae import ParameterDAE
from ...multilevel_object import local_object_factory
from ...variables import NullVariables
from . import AutomaticLocalConstraint
from . import Constraint


class ParameterLQRConstraint(Constraint):
    def __init__(self, higher_level_variables, local_level_variables,
                 parameter_dae):
        lower_level_variables = NullVariables()
        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables)
        if not isinstance(parameter_dae, ParameterDAE):
            raise TypeError(parameter_dae)
        self._parameter_dae = parameter_dae


class AutomaticLocalLQRConstraint(AutomaticLocalConstraint):
    pass


local_object_factory.register_localizer(ParameterLQRConstraint,
                                        AutomaticLocalLQRConstraint)

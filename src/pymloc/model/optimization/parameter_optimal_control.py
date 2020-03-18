import numpy as np
import scipy.integrate

from ..multilevel_object import MultiLevelObject
from ..multilevel_object import local_object_factory
from ..variables import InputStateVariables
from ..variables import NullVariables
from ..variables import ParameterContainer
from .constraints.parameter_lqr import ParameterLQRConstraint
from .objectives import Objective
from .objectives.parameter_lqr import ParameterLQRObjective
from .optimal_control import LQOptimalControl
from .optimization import AutomaticLocalOptimizationObject
from .optimization import OptimizationObject


class ParameterDependentOptimalControl(OptimizationObject):
    _local_object_class = LQOptimalControl

    def __init__(self, parameters: ParameterContainer,
                 state_input: InputStateVariables,
                 objective: ParameterLQRObjective,
                 constraint: ParameterLQRConstraint):
        if not isinstance(parameters, ParameterContainer):
            raise TypeError(parameters)
        self._parameters = parameters
        if not isinstance(state_input, InputStateVariables):
            raise TypeError(parameters)
        self._state_input = state_input
        lower_level_variables = NullVariables()
        local_level_variables = state_input
        higher_level_variables = parameters
        self._constraint = constraint
        self._objective = objective
        if not isinstance(objective, ParameterLQRObjective):
            raise TypeError(objective)
        if not isinstance(constraint, ParameterLQRConstraint):
            raise TypeError(constraint)
        super().__init__(self.objective, self.constraint,
                         lower_level_variables, higher_level_variables,
                         local_level_variables)

    @property
    def parameters(self):
        return self._parameters

    @property
    def state_input(self):
        return self._state_input

    @property
    def objective(self):
        return self._objective

    @property
    def constraint(self):
        return self._constraint


class AutomaticLQOptimalControl(AutomaticLocalOptimizationObject,
                                LQOptimalControl):
    def __init__(self, global_object, **kwargs):
        super().__init__(global_object, **kwargs)
        LQOptimalControl._init_local(self)


local_object_factory.register_localizer(ParameterDependentOptimalControl,
                                        AutomaticLQOptimalControl)

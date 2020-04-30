#
# Copyright (c) 2019-2020
#
# @author: Daniel Bankmann
# @company: Technische Universität Berlin
#
# This file is part of the python package pymloc
# (see https://gitlab.tubit.tu-berlin.de/bankmann91/python-mloc )
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#
from ...control_system.parameter_dae import LinearParameterControlSystem
from ...multilevel_object import local_object_factory
from ...variables import NullVariables
from . import AutomaticLocalConstraint
from . import Constraint
from .lqr import LQRConstraint


class ParameterLQRConstraint(Constraint):
    def __init__(self, higher_level_variables, local_level_variables,
                 parameter_dae_control, initial_value):
        lower_level_variables = NullVariables()
        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables)
        if not isinstance(parameter_dae_control, LinearParameterControlSystem):
            raise TypeError(parameter_dae_control)
        self._parameter_dae = parameter_dae_control

        self._dae = parameter_dae_control
        self._initial_value = initial_value

    @property
    def control_system(self):
        return self._dae

    @property
    def initial_value(self):
        return self._initial_value


class AutomaticLocalLQRConstraint(AutomaticLocalConstraint, LQRConstraint):
    def __init__(self, global_object, *args, **kwargs):
        self._global_object = global_object
        local_control_dae = global_object.control_system.get_localized_object()
        local_initial_value = global_object.localize_method(
            global_object.initial_value)
        LQRConstraint.__init__(self, local_control_dae, local_initial_value)


local_object_factory.register_localizer(ParameterLQRConstraint,
                                        AutomaticLocalLQRConstraint)

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
from abc import ABC

from ..multilevel_object import MultiLevelObject
from ..multilevel_object import local_object_factory
from ..variables.container import VariablesContainer
from .constraints.constraint import Constraint
from .local_optimization import LocalNullOptimization
from .local_optimization import LocalOptimizationObject
from .objectives.objective import Objective


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
        self.objective_object = objective_obj
        self.constraint_object = constraint_obj
        super().__init__(lower_level_variables, higher_level_variables,
                         local_level_variables)

    @property
    def objective_object(self) -> Objective:
        return self._objective_object

    @objective_object.setter
    def objective_object(self, value):
        self._objective_object = value

    @property
    def constraint_object(self) -> Constraint:
        return self._constraint_object

    @constraint_object.setter
    def constraint_object(self, value):
        self._constraint_object = value

    def get_sensitivities(self):
        raise NotImplementedError


class NullOptimization(OptimizationObject):
    _local_object_class = LocalNullOptimization


class AutomaticLocalOptimizationObject(LocalOptimizationObject):
    _auto_generated = True

    def __init__(self, global_optimization, hl_value=None, ll_value=None):
        self._global_object = global_optimization
        hl_vars = global_optimization.higher_level_variables
        if hl_value is not None:
            hl_vars.current_values = hl_value
        ll_vars = global_optimization.lower_level_variables
        if ll_value is not None:
            ll_vars.current_values = ll_value
        loc_objective = global_optimization.objective_object.get_localized_object(
        )
        loc_constraint = global_optimization.constraint_object.get_localized_object(
        )
        loc_vars = global_optimization.local_level_variables
        super().__init__(loc_objective, loc_constraint, loc_vars)


local_object_factory.register_localizer(NullOptimization,
                                        AutomaticLocalOptimizationObject)

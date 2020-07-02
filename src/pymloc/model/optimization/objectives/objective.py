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
import numpy as np

from ...multilevel_object import MultiLevelObject
from ...multilevel_object import local_object_factory
from ...variables import VariablesContainer
from .local_objective import LocalObjective


class Objective(MultiLevelObject):
    def residual(self, ll_vars: VariablesContainer,
                 hl_vars: VariablesContainer,
                 loc_vars: VariablesContainer) -> np.ndarray:
        ...


class AutomaticLocalObjective(LocalObjective):
    def __init__(self, global_objective, *args, **kwargs):
        self._global_objective = global_objective
        self.residual = self._global_objective.localize_method(
            self._global_objective.residual)
        super().__init__(*args, **kwargs)


local_object_factory.register_localizer(Objective, AutomaticLocalObjective)

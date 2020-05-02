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
import pytest

from pymloc.model.optimization.local_optimization import LocalOptimizationObject
from pymloc.model.optimization.optimization import OptimizationObject

from ..test_multilevelobject import MultiLevelObject


class TestOptimizationObject(MultiLevelObject):
    @pytest.fixture(autouse=True)
    def set_ml_object(self, opt):
        self.ml_object = opt[0]

    def test_contraint_function(self, opt):
        opt[0].constraint_object

    def test_opt_types(self, opt):
        variables, constraint, objective = opt[1:]
        with pytest.raises(TypeError):
            OptimizationObject(constraint, constraint, constraint, variables,
                               variables)
        with pytest.raises(TypeError):
            OptimizationObject(objective, objective, objective, variables,
                               variables)
        with pytest.raises(TypeError):
            OptimizationObject(objective, objective, variables, variables,
                               constraint)

    def test_localize(self, opt):
        opt_object = opt[0]
        local_opt_object = opt_object.get_localized_object()
        assert isinstance(local_opt_object, LocalOptimizationObject)

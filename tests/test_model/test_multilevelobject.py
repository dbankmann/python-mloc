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
from pymloc.model.variables.container import VariablesContainer
from pymloc.model.multilevel_object import local_object_factory
import pytest


class MultiLevelObject:
    def test_variables(self):
        vars = self.ml_object.lower_level_variables, self.ml_object.higher_level_variables, self.ml_object.local_level_variables
        for variables in vars:
            assert isinstance(variables, VariablesContainer)


class TestLocalObjectFactory:
    def test_register_function(self):
        with pytest.raises(ValueError):
            local_object_factory.register_localizer(object, "")

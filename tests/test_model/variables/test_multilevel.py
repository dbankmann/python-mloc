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
from pymloc.model.variables.multilevel import MultiLevelVariables
from pymloc.model.variables.time_function import StateVariables
import pytest


class TestMultiLevelVariables(object):
    @pytest.fixture
    def mlvars(self):
        return MultiLevelVariables()

    def test_add_variable_level(self, mlvars):
        variables = StateVariables(4)
        mlvars.add_variable_level(variables)

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
from pymloc.model.variables.time_function import InputVariables
from pymloc.model.variables.time_function import OutputVariables
from pymloc.model.variables.time_function import StateVariables


class Variables():
    def test_dimension(self):
        assert self.variables.dimension == 4
        assert self.variables.dimension != 5


class TimeVariables(Variables):
    def test_time_domain(self):
        assert self.variables.time_domain == [0., 1.]


class TestStateVariables(TimeVariables):
    variables = StateVariables(4)


class TestOutputVariables(TimeVariables):
    variables = OutputVariables(4)


class TestInputVariables(TimeVariables):
    variables = InputVariables(4)

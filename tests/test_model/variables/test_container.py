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
from pymloc.model.variables.container import InputOutputStateVariables


class TestInputOutputStateVariables:
    def test_state_system(self):
        vars = InputOutputStateVariables(6, 3, 2)
        assert vars.states == vars.variables[0]

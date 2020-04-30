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


class TestInitialValueProblem:
    def test_solve(self, initial_value_problem):
        initial_value_problem.init_solver(stepsize=1e-3)
        initial_value_problem.solve()

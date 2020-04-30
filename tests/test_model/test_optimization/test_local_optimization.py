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

from pymloc.model.optimization.constraints.local_constraint import LocalNullConstraint
from pymloc.model.optimization.local_optimization import LocalNullOptimization
from pymloc.model.optimization.objectives.local_objective import LocalNullObjective
from pymloc.model.variables.container import InputOutputStateVariables


class TestLocalOptimizationObject:
    @pytest.fixture
    def loc_opt(self):
        constraint = LocalNullConstraint()
        objective = LocalNullObjective()
        variables = InputOutputStateVariables(5, 4, 3)
        return LocalNullOptimization(objective, constraint, variables)

    def test_contraint_function(self, loc_opt):
        loc_opt.constraint

    def test_init_solver(self, loc_opt, *args, **kwargs):
        loc_opt.init_solver(*args, **kwargs)

    def test_set_default_solver(self, loc_opt):
        loc_opt.set_default_solver()

    def test_solve(self, loc_opt, *args, **kwargs):
        with pytest.raises(AttributeError):
            loc_opt.solve(*args, **kwargs)

    def test_solve2(self, loc_opt, *args, **kwargs):
        loc_opt.init_solver(**kwargs)
        sol = loc_opt.solve(*args)
        return sol

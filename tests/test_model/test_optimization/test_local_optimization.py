import pytest
from pymloc.model.optimization.local_optimization import LocalNullOptimization
from pymloc.model.optimization.constraints.local_constraint import LocalConstraint
from pymloc.model.optimization.objectives.local_objective import LocalObjective
from pymloc.model.variables.container import InputOutputStateVariables


class TestLocalOptimizationObject(object):
    @pytest.fixture
    def loc_opt(self):
        constraint = LocalConstraint()
        objective = LocalObjective()
        variables = InputOutputStateVariables(5, 4, 3)
        return LocalNullOptimization(objective, constraint, variables)

    def test_contraint_function(self, loc_opt):
        loc_opt.loc_constraint_object

    def test_init_solver(self, loc_opt):
        loc_opt.init_solver()

    def test_set_default_solver(self, loc_opt):
        loc_opt.set_default_solver()

    def test_solve(self, loc_opt):
        with pytest.raises(ValueError):
            loc_opt.solve()

    def test_solve2(self, loc_opt):
        loc_opt.init_solver()
        loc_opt.solve()

import pytest
from pymloc.model.optimization.local_optimization import LocalOptimizationObject
from pymloc.model.optimization.constraints.local_constraint import LocalConstraint
from pymloc.model.optimization.objectives.local_objective import LocalObjective
from pymloc.model.variables.container import InputOutputStateVariables


class TestLocalOptimizationObject(object):
    @pytest.fixture
    def loc_opt(self):
        constraint = LocalConstraint()
        objective = LocalObjective()
        variables = InputOutputStateVariables(5, 4, 3)
        return LocalOptimizationObject(objective, constraint, variables)

    def test_contraint_function(self, loc_opt):
        loc_opt.loc_constraint_object

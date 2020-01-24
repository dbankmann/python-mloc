import pytest
from pymloc.model.optimization.local_optimization import LocalOptimizationObject
from pymloc.model.optimization.constraints.local_constraint import LocalConstraint
from pymloc.model.optimization.objectives.local_objective import LocalObjective


class TestLocalOptimizationObject(object):
    @pytest.fixture
    def loc_opt(self):
        constraint = LocalConstraint()
        objective = LocalObjective()
        return LocalOptimizationObject(constraint, objective)

    def test_contraint_function(self, loc_opt):
        loc_opt.loc_constraint_object

from pymloc.model.optimization.constraints.constraint import Constraint
from pymloc.model.optimization.objectives.objective import Objective
from pymloc.model.optimization.optimization import OptimizationObject
from pymloc.model.variables.variables import Variables
import pytest

class TestOptimizationObject(object):


    @pytest.fixture
    def opt(self):
        constraint = Constraint()
        objective = Objective()
        variables = Variables()
        return OptimizationObject(objective, constraint, variables)



    def test_contraint_function(self, opt):
        opt.constraint_object

    def test_opt_types(self):
        constraint = Constraint()
        objective = Objective()
        variable = Variables()
        with pytest.raises(TypeError):
            OptimizationObject(constraint, constraint, constraint)
        with pytest.raises(TypeError):
            OptimizationObject(objective, objective, objective)
        with pytest.raises(TypeError):
            OptimizationObject(objective, objective, variable)

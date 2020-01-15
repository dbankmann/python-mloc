import pytest
from pymloc.mloc import MultiLevelOptimalControl
from pymloc.model.optimization.optimization import NullOptimization
from pymloc.solver.solver import NullSolver

@pytest.mark.incremental
class TestCreationOptimization(object):

    @pytest.mark.run(order=1)
    def test_creation_optimizations(self):
        optimizations = [NullOptimization(), NullOptimization()]
        solvers = [NullSolver(), NullSolver()]
        self.mloc = MultiLevelOptimalControl(optimizations, solvers)
        print("hi")

    @pytest.mark.run(order=2)
    def test_creation_bilevel(self):
        assert self.mloc.is_bilevel
        print("hi2")

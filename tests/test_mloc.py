import pytest
from pymloc.mloc import MultiLevelOptimalControl
from pymloc.model.optimization.optimization import NullOptimization
from pymloc.solver.solver import NullSolver

@pytest.mark.incremental
class TestCreationOptimization(object):

    @pytest.fixture
    def mloc_object(self):
        optimizations = [NullOptimization(), NullOptimization()]
        solvers = [NullSolver(), NullSolver()]
        mloc = MultiLevelOptimalControl(optimizations, solvers)
        return mloc

    def test_mloc_object(self, mloc_object):
        pass

    def test_creation_bilevel(self, mloc_object):
        assert mloc_object.levels == 2
        assert mloc_object.is_bilevel

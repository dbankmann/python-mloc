import pytest
from pymloc.model.optimization.local_optimization import LocalOptimizationObject


class TestLocalOptimizationObject(object):

    @pytest.fixture
    def loc_opt(self):
        return LocalOptimizationObject()

    def test_contraint_function(self, loc_opt):
        loc_opt.constraint_object

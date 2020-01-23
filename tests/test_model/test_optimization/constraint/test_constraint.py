from pymloc.model.optimization.constraints.constraint import Constraint
from ...test_multilevelobject import MultiLevelObject
import pytest


class TestConstraint(MultiLevelObject):
    @pytest.fixture(autouse=True)
    def set_ml_object(self, opt):
        self.ml_object = opt[2]

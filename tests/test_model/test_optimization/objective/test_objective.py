from pymloc.model.optimization.objectives.objective import Objective
from ...test_multilevelobject import MultiLevelObject
import pytest


class TestObjective(MultiLevelObject):
    @pytest.fixture(autouse=True)
    def set_ml_object(self, opt):
        self.ml_object = opt[3]

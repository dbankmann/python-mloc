from pymloc.model.variables.multilevel import MultiLevelVariables
from pymloc.model.variables.time_function import StateVariables
import pytest


class TestMultiLevelVariables(object):
    @pytest.fixture
    def mlvars(self):
        return MultiLevelVariables()

    def test_add_variable_level(self, mlvars):
        variables = StateVariables(4)
        mlvars.add_variable_level(variables)

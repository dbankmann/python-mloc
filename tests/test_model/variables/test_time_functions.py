from pymloc.model.variables.time_function import StateVariables
import pytest


class TestFiniteDimensionalTimeVariables(object):

    @pytest.fixture
    def fdt_variables(self):
        return StateVariables(dimension=4)

    def test_dimension(self, fdt_variables):
        assert fdt_variables.dimension == 4
        assert fdt_variables.dimension != 5

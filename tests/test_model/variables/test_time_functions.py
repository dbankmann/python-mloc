from pymloc.model.variables.time_function import StateVariables, OutputVariables, InputVariables
import pytest


class Variables():
    def test_dimension(self):
        assert self.variables.dimension == 4
        assert self.variables.dimension != 5


class TestStateVariables(Variables):
    variables = StateVariables(4)


class TestOutputVariables(Variables):
    variables = OutputVariables(4)


class TestInputVariables(Variables):
    variables = InputVariables(4)

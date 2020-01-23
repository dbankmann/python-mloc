from pymloc.model.variables.time_function import StateVariables, OutputVariables, InputVariables
import pytest


class Variables():
    def test_dimension(self):
        assert self.variables.dimension == 4
        assert self.variables.dimension != 5

class TimeVariables(Variables):
    def test_time_domain(self):
        assert self.variables.time_domain == [0., 1.]


class TestStateVariables(TimeVariables):
    variables = StateVariables(4)


class TestOutputVariables(TimeVariables):
    variables = OutputVariables(4)


class TestInputVariables(TimeVariables):
    variables = InputVariables(4)

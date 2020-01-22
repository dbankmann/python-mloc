from pymloc.model.variables.container import InputOutputStateVariables


class TestInputOutputStateVariables():
    def test_state_system(self):
        vars = InputOutputStateVariables(6, 3, 2)
        assert vars.states == vars.variables[0]

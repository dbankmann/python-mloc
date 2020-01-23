from pymloc.model.variables.container import VariablesContainer



class MultiLevelObject:
    def test_variables(self):
        vars = self.ml_object.lower_level_variables, self.ml_object.higher_level_variables, self.ml_object.local_level_variables
        for variables in vars:
            assert isinstance(variables, VariablesContainer)

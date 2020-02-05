from pymloc.model.variables.container import VariablesContainer
from pymloc.model.multilevel_object import local_object_factory
import pytest


class MultiLevelObject:
    def test_variables(self):
        vars = self.ml_object.lower_level_variables, self.ml_object.higher_level_variables, self.ml_object.local_level_variables
        for variables in vars:
            assert isinstance(variables, VariablesContainer)


class TestLocalObjectFactory:
    def test_register_function(self):
        with pytest.raises(ValueError):
            local_object_factory.register_localizer(object, "")

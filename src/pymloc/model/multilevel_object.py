from ..variables.container import VariablesContainer
from abc import ABC, abstractmethod


class MultiLevelObject(ABC):
    def __init__(self, lower_level_variables: VariablesContainer,
                 higher_level_variables: VariablesContainer,
                 local_level_variables: VariablesContainer):
        self._higher_level_variables = higher_level_variables
        self._lower_level_variables = lower_level_variables
        self._local_level_variables = local_level_variables
        for container in lower_level_variables, higher_level_variables, local_level_variables:
            if not isinstance(container, VariablesContainer):
                raise TypeError(container)

    @property
    def lower_level_variables(self):
        return self._lower_level_variables

    @property
    def higher_level_variables(self):
        return self._higher_level_variables

    @property
    def local_level_variables(self):
        return self._local_level_variables

    def localize_method(self, method):
        def localized_function(variables):
            hl_vars = self.higher_level_variables
            ll_vars = self.lower_level_variables
            return method(hl_vars, variables, ll_vars)

        return localized_function

    def get_localized_object(self, *args, **kwargs):
        return local_object_factory.get_localized_object(self, *args, **kwargs)


class LocalObjectFactory:
    def __init__(self):
        self._localizer_objects = {}

    def register_localizer(self, global_object, localizer_object):
        if not issubclass(global_object, MultiLevelObject):
            raise ValueError(global_object)
        self._localizer_objects[global_object] = localizer_object

    def get_localized_object(self, global_object, *args, **kwargs):
        localizer_object = self._localizer_objects.get(global_object.__class__)
        if not localizer_object:
            raise ValueError(global_object)
        return localizer_object(global_object, *args, **kwargs)


local_object_factory = LocalObjectFactory()

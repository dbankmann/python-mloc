from ..variables.container import VariablesContainer
from abc import ABC, abstractmethod
class MultiLevelObject(ABC):
    def __init__(self, lower_level_variables: VariablesContainer, higher_level_variables: VariablesContainer, local_level_variables: VariablesContainer):
        self._higher_level_variables = higher_level_variables
        self._lower_level_variables = lower_level_variables
        self._local_level_variables = local_level_variables
    @property
    def lower_level_variables(self):
        return self._lower_level_variables

    @property
    def higher_level_variables(self):
        return self._higher_level_variables

    @property
    def local_level_variables(self):
        return self._local_level_variables


    @abstractmethod
    def get_localized_object(self):
        pass

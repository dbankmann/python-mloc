from .variables import Variables

from abc import ABC, abstractmethod


class TimeVariables(Variables, ABC):
    def __init__(self, dimension, time_domain):
        self._time_domain = time_domain
        super().__init__(dimension)

    @property
    def time_domain(self):
        return self._time_domain



class StateVariables(TimeVariables):
    def __init__(self, dimension, time_domain=[0.,1.]):
        super().__init__(dimension, time_domain)



class InputVariables(TimeVariables):
    def __init__(self, dimension, time_domain=[0.,1.]):
        super().__init__(dimension, time_domain)


class OutputVariables(TimeVariables):
    def __init__(self, dimension, time_domain=[0.,1.]):
        super().__init__(dimension, time_domain)

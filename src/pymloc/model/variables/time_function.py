from .variables import Variables

from abc import ABC


class TimeVariables(Variables, ABC):
    def __init__(self, time_domain):
        self.time_domain = time_domain

class FiniteDimensionalTimeVariables(TimeVariables, ABC):
    def __init__(self, dimension, *args, **kwargs):
        self._dimension = dimension


    @property
    def dimension(self):
        return self._dimension

    @dimension.setter
    def dimension(self, value):
        if value < 0:
            raise ValueError("Dimensions have to be non-negative")
        self._dimension = value


class StateVariables(FiniteDimensionalTimeVariables):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

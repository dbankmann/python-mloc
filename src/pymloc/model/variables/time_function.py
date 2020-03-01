from abc import ABC
from abc import abstractmethod

import numpy as np

from .variables import Variables


class TimeVariables(Variables, ABC):
    def __init__(self, dimension, time_domain):
        self._time_domain = time_domain
        self._current_time = time_domain[0]
        super().__init__(dimension)

    @property
    def time_domain(self):
        return self._time_domain

    @property
    def current_values(self):
        return self._current_values

    @current_values.setter
    def current_values(self, value):
        self._current_values = value

    @property
    def current_time(self):
        return self._current_time

    @current_time.setter
    def current_time(self, value):
        self._current_time = value

    @property
    def shape(self):
        return np.empty(self.dimension).shape


class StateVariables(TimeVariables):
    def __init__(self, dimension, time_domain=[0., 1.]):
        super().__init__(dimension, time_domain)


class InputVariables(TimeVariables):
    def __init__(self, dimension, time_domain=[0., 1.]):
        super().__init__(dimension, time_domain)


class OutputVariables(TimeVariables):
    def __init__(self, dimension, time_domain=[0., 1.]):
        super().__init__(dimension, time_domain)


class Time(TimeVariables):
    def __init__(self, t_0, t_f):
        self.t_0 = t_0
        self.t_f = t_f
        super().__init__(dimension=1, time_domain=[t_0, t_f])
        self._grid = None

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value

    def get_random_values(self):
        return np.random.random(1)[0]

from abc import ABC, abstractmethod
import numpy as np


class Variables(ABC):
    def __init__(self, dimension):
        self._dimension = dimension
        self._current_values = None

    @property
    def dimension(self):
        return self._dimension

    @dimension.setter
    def dimension(self, value):
        self._dimension = value

    @property
    @abstractmethod
    def current_values(self):
        pass

    @current_values.setter
    @abstractmethod
    def current_values(self, value):
        pass

    def get_random_values(self):
        return np.random.random(self.dimension)

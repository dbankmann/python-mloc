from abc import ABC, abstractmethod


class Variables(ABC):
    def __init__(self, dimension):
        self._dimension = dimension

    @property
    def dimension(self):
        return self._dimension

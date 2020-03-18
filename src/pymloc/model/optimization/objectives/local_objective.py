from abc import ABC
from abc import abstractmethod


class LocalObjective(ABC):
    def __init__(self):
        pass


class LocalNullObjective(LocalObjective):
    def value(self):
        pass

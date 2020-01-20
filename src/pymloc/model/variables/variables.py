from abc import ABC


class Variables(ABC):


    @property
    def vars(self):
        return self._vars


    def add_variables(self, variables):
        pass

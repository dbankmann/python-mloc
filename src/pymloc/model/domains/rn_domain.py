from abc import ABC, abstractmethod


class Domain(ABC):
    def __init__(self):
        pass

    def in_domain(self, variable):
        try:
            for value in variable.current_values:
                assert self._in_domain_assertion(value)
            return True
        except AssertionError:
            #TODO: Can be softened to only raise a warning
            raise ValueError("Variable value out of domain")

    @abstractmethod
    def _in_domain_assertion(self):
        pass


class RNDomain(Domain):
    def __init__(self, dimension):
        self._dimension = dimension

    def _in_domain_assertion(self, value):
        return True

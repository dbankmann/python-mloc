from .variables import Variables


class Parameters(Variables):
    def __init__(self, dimension, domain):
        self._domain = domain
        super().__init__(dimension)

    @property
    def current_values(self):
        vals = self._current_values
        if vals is not None and vals.size == 1:
            return vals.item()
        else:
            return vals

    @current_values.setter
    def current_values(self, value):
        self._current_values = value

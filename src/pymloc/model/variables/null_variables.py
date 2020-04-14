from .container import VariablesContainer


class NullVariables(VariablesContainer):
    def __init__(self):
        super().__init__()
        self._associated_problem = None

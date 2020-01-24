from abc import ABC


class Variable(ABC):
    pass


class TimeVariable(Variable, ABC):
    pass


class Input(TimeVariable):
    pass


class Output(TimeVariable):
    pass


class State(TimeVariable):
    pass

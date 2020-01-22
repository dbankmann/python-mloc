from .variables import Variables

from abc import ABC, abstractmethod


class TimeVariables(Variables, ABC):
    def __init__(self, *args, **kwargs):
        self.time_domain = None
        super().__init__(*args, **kwargs)




class StateVariables(TimeVariables):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



class InputVariables(TimeVariables):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class OutputVariables(TimeVariables):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

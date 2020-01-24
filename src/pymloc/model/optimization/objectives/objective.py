from ..multilevel_object import MultiLevelObject
from .local_objective import LocalObjective


class Objective(MultiLevelObject):
    def get_localized_object(self):
        return LocalObjective()

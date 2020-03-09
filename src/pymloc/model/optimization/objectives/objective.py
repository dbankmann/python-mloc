from ...multilevel_object import MultiLevelObject
from ...multilevel_object import local_object_factory
from .local_objective import LocalObjective


class Objective(MultiLevelObject):
    def residual(self):
        pass


class AutomaticLocalObjective(LocalObjective):
    def __init__(self, global_objective, *args, **kwargs):
        self._global_objective = global_objective
        self.residual = self._global_objective.localize_method(
            self._global_objective.residual)
        super().__init__(*args, **kwargs)


local_object_factory.register_localizer(Objective, AutomaticLocalObjective)

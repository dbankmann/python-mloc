from ..multilevel_object import MultiLevelObject
from .local_constraint import LocalConstraint
class Constraint(MultiLevelObject):
    def get_localized_object(self):
        return LocalConstraint()

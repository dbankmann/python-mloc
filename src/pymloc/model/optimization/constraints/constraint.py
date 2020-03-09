from ...multilevel_object import MultiLevelObject, local_object_factory
from .local_constraint import LocalConstraint


class Constraint(MultiLevelObject):
    pass


class AutomaticLocalConstraint(LocalConstraint):
    def __init__(self, global_constraint, *args, **kwargs):
        self._global_constraint = global_constraint
        super().__init__(*args, **kwargs)


local_object_factory.register_localizer(Constraint, AutomaticLocalConstraint)


class NullConstraint(Constraint):
    pass


class AutomaticLocalNullConstraint(AutomaticLocalConstraint):
    pass


local_object_factory.register_localizer(NullConstraint,
                                        AutomaticLocalNullConstraint)

from . import LocalConstraint


class LQRConstraint(LocalConstraint):
    def __init__(self, dae_control, initial_value):
        self._dae = dae_control
        self._initial_value = initial_value

    @property
    def control_system(self):
        return self._dae

    @property
    def initial_value(self):
        return self._initial_value

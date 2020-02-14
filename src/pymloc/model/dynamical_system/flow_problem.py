from .. import Solvable


class LinearFlow(Solvable):
    def __init__(self, time_interval, flow_dae):
        super().__init__()
        self._flow_dae = flow_dae
        self._nn = flow_dae._nn

    @property
    def flow_dae(self):
        return self._flow_dae

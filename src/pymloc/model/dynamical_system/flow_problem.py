#
# Copyright (c) 2019-2020
#
# @author: Daniel Bankmann
# @company: Technische Universität Berlin
#
# This file is part of the python package pymloc
# (see https://gitlab.tubit.tu-berlin.de/bankmann91/python-mloc )
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#
from .. import Solvable
from .representations import LinearFlowRepresentation


class LinearFlow(Solvable):
    def __init__(self, time_interval, flow_dae):
        super().__init__()
        if not isinstance(flow_dae, LinearFlowRepresentation):
            raise TypeError(
                "dae object needs to be in LinearFlowRepresentation")
        self._flow_dae = flow_dae
        self._nn = flow_dae._nn

    @property
    def flow_dae(self):
        return self._flow_dae

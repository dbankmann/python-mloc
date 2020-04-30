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
from . import LocalConstraint


class LQRConstraint(LocalConstraint):
    def __init__(self, dae_control, initial_value):
        self._dae = dae_control
        self._initial_value = initial_value

    def reset(self):
        self._dae.reset()

    @property
    def control_system(self):
        return self._dae

    @property
    def initial_value(self):
        return self._initial_value

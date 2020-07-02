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
from abc import ABC

import numpy as np


class LocalObjective(ABC):
    """Base class for all localized objectives."""

    # TODO: Create interface
    def __init__(self):
        ...

    def value(self, solution) -> np.ndarray:
        ...


class LocalNullObjective(LocalObjective):
    def value(self):
        return np.zeros((0))

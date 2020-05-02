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
from abc import abstractmethod

import numpy as np


class Variables(ABC):
    def __init__(self, dimension):
        self._dimension = dimension
        self._current_values = None

    @property
    def dimension(self):
        return self._dimension

    @dimension.setter
    def dimension(self, value):
        self._dimension = value

    @abstractmethod
    def current_values(self):
        pass

    def get_random_values(self):
        return np.random.random(self.dimension)

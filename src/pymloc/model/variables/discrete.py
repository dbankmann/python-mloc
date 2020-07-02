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
import numpy as np

import pymloc

from .variables import Variables


class Parameters(Variables):
    """Parameter variables."""
    def __init__(self, dimension, domain: 'pymloc.model.domains.RNDomain'):
        self._domain = domain
        super().__init__(dimension)

    @property
    def current_values(self) -> np.ndarray:
        vals = self._current_values
        if isinstance(vals, float):
            return np.float(vals)
        elif vals is not None and vals.size == 1:
            return vals.item()
        else:
            return vals

    @current_values.setter
    def current_values(self, value):
        self._current_values = value

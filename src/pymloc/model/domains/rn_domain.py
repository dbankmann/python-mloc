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

from ..variables import VariablesContainer


class Domain(ABC):
    """Baseclass for all domains."""
    def __init__(self):
        pass

    def in_domain(self, variable: VariablesContainer) -> bool:
        try:
            for value in variable.current_values:
                assert self._in_domain_assertion(value)
            return True
        except AssertionError:
            # TODO: Can be softened to only raise a warning
            raise ValueError("Variable value out of domain")

    @abstractmethod
    def _in_domain_assertion(self, value: np.ndarray) -> bool:
        pass


class RNDomain(Domain):
    r"""Domain :math:`\mathbb R^n`"""
    def __init__(self, dimension: int):
        self._dimension = dimension

    def _in_domain_assertion(self, value):
        return True

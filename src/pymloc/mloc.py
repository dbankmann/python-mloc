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

from .model import Solvable


"""
Definition of multilevel optimization problems as in thesis.
We assume that the constraint function G_i are equal for all i.
"""


class MultiLevelOptimalControl(Solvable):
    """Driver routine that initializes and handles all the solvers.
    """
    def __init__(self, optimizations, variables):
        self.optimizations = optimizations
        self.levels = len(self.optimizations)
        assert self.levels > 0
        self.variables = variables
        assert len(variables) == self.levels
        if self.levels == 2:
            self.is_bilevel = True
        else:
            self.is_bilevel = False
        super().__init__()

    @property
    def lowest_opt(self):
        return self.optimizations[-1]

    @property
    def highest_opt(self):
        return self.optimizations[0]

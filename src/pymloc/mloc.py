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

from typing import List

from .model import Solvable
from .model.optimization.optimization import OptimizationObject
from .model.variables.container import VariablesContainer


"""
Definition of multilevel optimization problems as in thesis.
"""


class MultiLevelOptimalControl(Solvable):
    """Main object for multilevel optimal control object. Initialization and handling of the
    respective solvers of the optimizations is left to the solver object of
    MultiLevelOptimalControl.
    """
    is_bilevel: bool

    def __init__(self, optimizations: List[OptimizationObject],
                 variables: List[VariablesContainer]):
        self.optimizations = optimizations
        self.levels: int = len(self.optimizations)
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

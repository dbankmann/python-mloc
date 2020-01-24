import numpy as np
"""
Definition of multilevel optimization problems as in thesis.
We assume that the constraint function G_i are equal for all i.
"""


class MultiLevelOptimalControl(object):
    """Driver routine that initializes and handles all the solvers.
    """
    def __init__(self, optimizations, solvers):
        self.optimizations = optimizations
        self.levels = len(self.optimizations)
        assert self.levels > 0
        self.solvers = solvers
        assert len(solvers) == self.levels
        if self.levels == 2:
            self.is_bilevel = True
        else:
            self.is_bilevel = False

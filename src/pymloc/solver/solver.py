import numpy as np


class BaseSolver(object):
    def __init__(self, model, abs_tol=1.e-3, allowed_error=1.e-3, max_iter=10):
        self.abs_tol = abs_tol
        self.model = model
        self.allowed_error = allowed_error
        self.max_iter = max_iter

    def solve(self, *args, **kwargs):
        raise NotImplementedError

    def output(self):
        raise NotImplementedError


class NullSolver(BaseSolver):
    def __init__(self):
        pass

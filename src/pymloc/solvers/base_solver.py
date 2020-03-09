from abc import ABC
from abc import abstractmethod

import numpy as np


class BaseSolver(ABC):
    def __init__(self, model, abs_tol=1.e-3, rel_tol=1.e-3, max_iter=10):
        self.abs_tol = abs_tol
        self.model = model
        self.rel_tol = rel_tol
        self.max_iter = max_iter

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def output(self):
        raise NotImplementedError

    def abort(self, residual):
        return np.allclose(residual, 0., atol=self.abs_tol, rtol=self.rel_tol)


class Solution:
    def __init__(self, solution, params=None):
        self._solution = solution
        self._solver_params = params

    @property
    def solution(self):
        return self._solution

    @property
    def params(self):
        return self._solver_params


class TimeSolution(Solution):
    def __init__(self, time_grid, solution):
        super().__init__(solution)
        self._time_grid = time_grid
        solution_time_dict = {
            time_grid[i]: solution[..., i]
            for i in range(time_grid.size)
        }
        self._solution_time_dict = solution_time_dict

    @property
    def time_grid(self):
        return self._time_grid

    def __call__(self, t):
        sol = self._solution_time_dict.get(t)
        if sol is None:
            raise ValueError("Time: {} not in time_grid".format(t))
        else:
            return sol

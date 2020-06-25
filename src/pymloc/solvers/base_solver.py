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
from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from typing import Dict

import numpy as np

from ..model.solvable import Solvable

logger = logging.getLogger(__name__)


class BaseSolver(ABC):
    """Base class for all solvers.
    Sets common attributes and handles running the solver.
    """
    @abstractmethod
    def __init__(self,
                 model: Solvable = None,
                 abs_tol: float = 1.e-3,
                 rel_tol: float = 1.e-3,
                 max_iter: int = 10):
        self.abs_tol = abs_tol
        self.model = model
        self.rel_tol = rel_tol
        self.max_iter = max_iter
        self._solver_params = dict({
            "abs_tol": abs_tol,
            "rel_tol": rel_tol,
            "max_iter": max_iter
        })

    def run(self, *args, **kwargs):
        logger = logging.getLogger(self.__class__.__module__)
        solver_level.level += 1
        logger.info("Starting solver {}".format(self.__class__.__name__))
        logger.info(
            "Current option values:\nabs_tol: {}\nrel_tol: {}\nmax_iter: {}".
            format(self.abs_tol, self.rel_tol, self.max_iter))
        sol = self._run(*args, **kwargs)
        solver_level.level -= 1
        return sol

    @abstractmethod
    def _run(self, *args, **kwargs):
        pass

    def output(self):
        raise NotImplementedError

    def abort(self, residual: np.ndarray) -> np.float:
        return np.allclose(residual, 0., atol=self.abs_tol, rtol=self.rel_tol)


class Solution:
    """Class for the solution of a solver run"""
    def __init__(self, solution: np.ndarray, params=None):
        self._solution = solution
        self._solver_params = params

    @property
    def solution(self):
        return self._solution

    @property
    def params(self):
        return self._solver_params


class TimeSolution(Solution):
    """Solution subclass that allows for solutions on a time-grid"""
    def __init__(self,
                 time_grid: np.ndarray,
                 solution: np.ndarray,
                 interpolation: bool = False,
                 dynamic_update=None,
                 params=None):
        super().__init__(solution, params)
        self._time_grid = time_grid
        # TODO: Make more efficient
        solution_time_dict = {
            time_grid[i]: solution[..., i]
            for i in range(time_grid.size)
        }
        self._solution_time_dict = solution_time_dict
        self._interpolation = interpolation
        self._dynamic_update = dynamic_update

        self._current_t: Dict[np.float, str] = dict()

    @property
    def dynamic_update(self):
        return self._dynamic_update

    @dynamic_update.setter
    def dynamic_update(self, value):
        self._dynamic_update = value

    @property
    def interpolation(self):
        return self._interpolation

    @interpolation.setter
    def interpolation(self, value):
        self._interpolation = value

    @property
    def time_grid(self):
        return self._time_grid

    def __call__(self, t: float):
        sol = self._solution_time_dict.get(t)
        if sol is None:
            if self._interpolation:
                self._recompute_interpolated(t)
                return self._current_interpolated
            elif self._dynamic_update is not None:
                return self._add_solution(t)
            else:
                raise ValueError("Time: {} not in time_grid".format(t))
        else:
            return sol

    def _add_solution(self, t):
        grid = self.time_grid
        idx = grid.searchsorted(t)
        time_grid = np.array([t])
        sol = self._dynamic_update(self, time_grid)
        if isinstance(sol, TimeSolution):
            sol = sol(t)
        self._time_grid = np.insert(grid, idx, t)
        self._solution = np.insert(self._solution, idx, sol, axis=-1)
        self._solution_time_dict[t] = sol
        return sol

    def _recompute_interpolated(self, t):
        if self._check_current_time(t, "interpolate"):
            logger.warning("Interpolating value...\nPotentially slow!")
            idx = np.searchsorted(self._time_grid,
                                  t)  # TODO: Improvable for sorted array?
            t0, tf = self._time_grid[idx - 1:idx + 1]
            m = tf - t0
            x = self._solution_time_dict.get
            sol = x(t0) + (t - t0) * (x(tf) - x(t0)) / m
            self._current_interpolated = sol

    def _check_current_time(self, t, method):
        if self._current_t.get(method) is None or self._current_t[method] != t:
            self._current_t[method] = t
            return True
        else:
            return False


class Level:
    """Class provides the current level of a solver inside the multilevel approach.
    Used by loggers to filter and format the output."""
    __instance = None

    @staticmethod
    def get_instance() -> None:
        if Level.__instance is None:
            Level()
        assert Level.__instance is not None
        return Level.__instance

    def __init__(self):
        if Level.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Level.__instance = self
        self.level = 0


solver_level = Level()

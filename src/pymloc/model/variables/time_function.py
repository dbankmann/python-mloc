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
import logging
from abc import ABC
from typing import Optional
from typing import Tuple

import numpy as np

from .variables import Variables

logger = logging.getLogger(__name__)


class TimeVariables(Variables, ABC):
    """Subclass for variables that depend on time."""
    def __init__(self, dimension: int, time_domain: np.ndarray):
        self._time_domain = time_domain
        self._current_time = time_domain[0]
        super().__init__(dimension)

    @property
    def time_domain(self) -> np.ndarray:
        return self._time_domain

    @property
    def current_values(self) -> np.ndarray:
        return self._current_values

    @current_values.setter
    def current_values(self, value):
        self._current_values = value

    @property
    def current_time(self) -> float:
        return self._current_time

    @current_time.setter
    def current_time(self, value):
        self._current_time = value

    @property
    def shape(self) -> Tuple[int, ...]:
        return np.empty(self.dimension).shape


class StateVariables(TimeVariables):
    def __init__(self, dimension, time_domain=[0., 1.]):
        super().__init__(dimension, time_domain)


class InputVariables(TimeVariables):
    def __init__(self, dimension, time_domain=[0., 1.]):
        super().__init__(dimension, time_domain)


class OutputVariables(TimeVariables):
    def __init__(self, dimension, time_domain=[0., 1.]):
        super().__init__(dimension, time_domain)


class Time(TimeVariables):
    """The time variable."""
    def __init__(self,
                 t_0: float,
                 t_f: float,
                 time_grid: Optional[np.ndarray] = None):
        self.t_0 = t_0
        self.t_f = t_f
        super().__init__(dimension=1, time_domain=[t_0, t_f])
        self.grid = time_grid

    def add_to_grid(self, tp: float):
        """Adds a timepoint tp to the current grid."""
        if not self.t_0 < tp < self.t_f:
            raise ValueError(tp)
        elif tp in self.grid:
            logger.warning("{} already in grid".format(tp))
            return
        idx = np.searchsorted(self.grid, tp)
        self.grid = np.insert(self.grid, idx, tp)

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    @grid.setter
    def grid(self, value):
        if value is None:
            self._grid = None
        else:
            if self.t_0 not in value:
                if np.allclose(self.t_0, value[0]):
                    value[0] = self.t_0
                else:
                    value = np.insert(value, 0, self.t_0)
            if self.t_f not in value:
                if np.allclose(self.t_f, value[-1]):
                    value[-1] = self.t_f
                else:
                    value = np.append(value, self.t_f)
            self._grid = value

    def at_bound(self, tau: float) -> bool:
        """Tests whether tau is at the upper or lower bound of time_grid"""
        return self.at_lower_bound(tau) or self.at_upper_bound(tau)

    def at_upper_bound(self, tau: float) -> bool:
        """Tests whether tau is at the upper bound of time_grid"""
        if self.t_f == tau:
            return True
        else:
            return False

    def at_lower_bound(self, tau: float) -> bool:
        """Tests whether tau is at the lower bound of time_grid"""
        if self.t_0 == tau:
            return True
        else:
            return False

    def get_random_values(self) -> float:
        return np.random.random(1)[0]

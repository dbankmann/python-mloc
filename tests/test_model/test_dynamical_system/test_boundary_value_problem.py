from pymloc.model.dynamical_system.boundary_value_problem import BoundaryValueProblem
from pymloc.model.variables.time_function import Time
import numpy as np
import pytest


class TestBoundaryValueProblem:
    def test_init(self, linear_dae):
        gamma_0 = np.diag((1., 2.))
        gamma_f = np.diag((0., 1.))
        gamma = np.array([1., 3.])
        time_interval = Time(0., 1.)
        BoundaryValueProblem(time_interval, gamma_0, gamma_f, gamma,
                             linear_dae)

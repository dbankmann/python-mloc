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
import pytest

from pymloc.model.dynamical_system.boundary_value_problem import BoundaryValueProblem
from pymloc.model.dynamical_system.boundary_value_problem import BoundaryValues
from pymloc.model.variables.time_function import Time


@pytest.fixture
def boundary_values():
    gamma_0 = np.diag((1., 2.))
    gamma_f = np.diag((0., 1.))
    gamma = np.array([1., 3.])
    return BoundaryValues(gamma_0, gamma_f, gamma)


@pytest.fixture
def bvp(linear_dae, boundary_values):
    time_interval = Time(0., 1.)
    return BoundaryValueProblem(time_interval, linear_dae, boundary_values)


class TestBoundaryValueProblem:
    def test_init(self, bvp):
        assert bvp.boundary_values.nnodes == 2

    def test_boundary_residual(self, bvp):
        node_values = np.array([[10., 100.], [1000., -10.]])
        residual = bvp.boundary_values.residual(node_values)
        assert np.allclose(
            residual,
            np.array([10. + 0., 2000. + -10.]) - np.array([1., 3.]))

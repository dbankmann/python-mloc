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

from pymloc.model.optimization.local_optimization import LocalNullOptimization
from pymloc.model.solvable import solver_container_factory
from pymloc.solvers import NullSolver


class TestSolvers:
    def test_set_default_solver(self, local_opt):
        solver_container = solver_container_factory.get_solver_container(
            local_opt[0])
        null_solver = solver_container.solvers[0]
        solver_container.default_solver = null_solver
        assert solver_container.default_solver.solver == NullSolver


class TestSolverContainerFactory:
    def test_solver_container_creation(self):
        solver = NullSolver
        opt = LocalNullOptimization
        solver_container_factory.register_solver(solver, opt)

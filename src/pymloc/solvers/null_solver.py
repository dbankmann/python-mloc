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
from ..model.optimization import local_optimization
from ..solver_container import solver_container_factory
from . import BaseSolver


class NullSolver(BaseSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _run(self, *args, **kwargs):
        pass


solver_container_factory.register_solver(
    local_optimization.LocalNullOptimization, NullSolver, default=True)

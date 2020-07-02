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
"""Contains the SolverContainer object, and solver_container_factory, which are used to maintain a mapping
between solvable models and their solvers."""
from .container import SolverContainer
from .container import SolverTuple
from .factory import solver_container_factory

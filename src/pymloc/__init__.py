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
"""
isort:skip_file
"""
__version__ = '0.0.1'

# Enable 64 bit on startup
import jax.config

# Models have to be imported before solvers, because of the coupling between them in solver_container_factory
from . import logger
from . import model
from .model import optimization
from .model import sensitivities
from .model import variables
from . import solver_container
from . import solvers
from . import level_filter
from .mloc import MultiLevelOptimalControl

jax.config.update("jax_enable_x64", True)

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

from .solvers.base_solver import solver_level


class LevelFilter(logging.Filter):
    """Adds indentation to every logger according to the current solver hierarchy."""
    max_level = 4

    def filter(self, record):
        level = solver_level.level

        indents = level * '\t'
        record.msg = record.msg.replace('\n', '\n' + indents)
        record.level_indent = indents
        return level < LevelFilter.max_level


rootlogger = logging.getLogger()
rootlogger.handlers[0].addFilter(LevelFilter())

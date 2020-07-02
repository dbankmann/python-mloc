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
from .container import VariablesContainer


class NullVariables(VariablesContainer):
    """Dummy variables."""
    def __init__(self):
        super().__init__()
        self._associated_problem = None

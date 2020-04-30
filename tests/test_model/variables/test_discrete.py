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
import pytest
from pymloc.model.variables import Parameters
from pymloc.model.domains import RNDomain


class TestParameters:
    def test_domain(self):
        Parameters(4, domain=RNDomain(4))

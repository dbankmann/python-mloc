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
from pymloc.model.domains import RNDomain
from pymloc.model.variables import Parameters


class TestRNDomain:
    def test_init_and_assertion(self):
        domain = RNDomain(4)
        params = Parameters(4, domain)
        domain.in_domain(params)

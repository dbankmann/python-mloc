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

from pymloc.mloc import MultiLevelOptimalControl
from pymloc.model.variables import NullVariables
from pymloc.solvers import NullSolver


@pytest.mark.incremental
class TestCreationOptimization(object):
    @pytest.fixture
    def mloc_object(self, opt):
        optimizations = [opt[0], opt[0]]
        variableslist = [NullVariables(), NullVariables()]
        mloc = MultiLevelOptimalControl(optimizations, variableslist)
        return mloc

    def test_mloc_object(self, mloc_object):
        pass

    def test_creation_bilevel(self, mloc_object):
        assert mloc_object.levels == 2
        assert mloc_object.is_bilevel

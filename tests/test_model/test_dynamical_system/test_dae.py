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


class TestLinearDAE:
    def test_t2(self, linear_real_dae):
        t2 = linear_real_dae.t2(0.)
        rank = linear_real_dae.rank
        n = 3
        assert t2.shape == (n, rank)

    def test_t2prime(self, linear_real_dae):
        t2prime = linear_real_dae.t2prime(0.)
        e0 = linear_real_dae.e(0.)
        assert t2prime.shape == (3, 1)
        assert np.allclose(e0 @ t2prime, [0.])

    def test_z1(self, linear_real_dae):
        t2 = linear_real_dae.z1(0.)
        assert t2.shape == (3, 2)

    def test_z1prime(self, linear_real_dae):
        t2prime = linear_real_dae.z1prime(0.)
        e0 = linear_real_dae.e(0.)
        assert np.allclose(t2prime.T @ e0, [0.])

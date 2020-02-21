import pytest


class TestLinearDAE:
    def test_t2(self, linear_real_dae):
        t2 = linear_real_dae.t2(0.)
        rank = linear_real_dae.rank
        n = 3
        assert t2.shape == (3, 2)

    def test_t2prime(self, linear_real_dae):
        t2prime = linear_real_dae.t2prime(0.)
        rank = linear_real_dae.rank
        n = 3
        assert t2prime.shape == (3, 1)

    def test_z1(self, linear_real_dae):
        t2 = linear_real_dae.z1(0.)
        rank = linear_real_dae.rank
        n = 3
        assert t2.shape == (3, 2)

    def test_z1prime(self, linear_real_dae):
        t2prime = linear_real_dae.z1prime(0.)
        rank = linear_real_dae.rank
        n = 3
        assert t2prime.shape == (3, 1)

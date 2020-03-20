import numpy as np
import pytest

from pymloc.model.dynamical_system.representations import LinearFlowRepresentation
from pymloc.model.variables.container import StateVariablesContainer

from .test_dae import TestLinearDAE


@pytest.fixture
def dae():
    def e(t):
        return np.array([[0., 1., 0.], [-1., 0., 0.], [-0., 0., 0.]])

    def a(t):
        return np.array([[0., -1., 1.], [-1., 3., 0.], [1., 0., 1.]])

    def f(t):
        return np.array([0., 0., 0.])

    variables = StateVariablesContainer(3)
    flow = LinearFlowRepresentation(variables, e, a, f, 3)
    flow.init_rank()
    return flow


class TestFlowRepresentation(TestLinearDAE):
    def test_projection_ehat(self, dae):
        proj_val = dae.ehat_1(0.)

        ref_val = np.array([[0., 1., 0.], [-1., 0., 0.]])
        assert np.allclose(proj_val, ref_val)

    def test_projection_ehat2(self, dae):
        proj_val = dae.ehat_2(0.)

        ref_val = np.zeros(3)
        assert np.allclose(proj_val, ref_val)

    def test_projection_ahat(self, dae):
        proj_val = dae.ahat_1(0.)

        ref_val = np.array([[0., -1., 1.], [-1., 3., 0.]])
        assert np.allclose(proj_val, ref_val)

    def test_projection_ahat2(self, dae):
        proj_val = dae.ahat_2(0.)
        ref_val = np.array([[1., 0., 1.]])
        assert np.allclose(proj_val, ref_val)

    def test_projection_fhat(self, dae):
        proj_val = dae.fhat_1(0.)

        ref_val = np.zeros(2)
        assert np.allclose(proj_val, ref_val)

    def test_projection_fhat2(self, dae):
        proj_val = dae.fhat_2(0.)
        ref_val = np.zeros(1)
        assert np.allclose(proj_val, ref_val)

    def test_projection(self, dae):
        proj_val = dae.projection(0.)

        ref_val = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]])

        assert np.allclose(proj_val, ref_val)

    def test_projection(self, dae):
        proj_val = dae.projection_complement(0.)

        ref_val = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]])

        assert np.allclose(proj_val, ref_val)

    def test_projection_der(self, dae):
        proj_val = dae.projection_derivative(0.)

        ref_val = np.zeros((3, 3))

        assert np.allclose(proj_val, ref_val)

    def test_projection_da(self, dae):
        proj_val = dae.d_a(0.)

        ref_val = np.array([[0., 0., 0.], [0., 0., 0.], [1., 0., 1.]])
        assert np.allclose(proj_val, ref_val)

    def test_projection_cal(self, dae):
        proj_val = dae.cal_projection(0.)

        ref_val = np.array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]])
        assert np.allclose(proj_val, ref_val)

    def test_projection_dd(self, dae):
        proj_val = dae.d_d(0.)

        ref_val = np.array([[1., -3., 0.], [-1., -1., 0.], [0., 0., 0.]])

        assert np.allclose(proj_val, ref_val)

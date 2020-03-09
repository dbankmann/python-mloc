import pytest

from pymloc.model.optimization.nonlinear_leastsquares import NonLinearLeastSquares


@pytest.fixture
def nllq(variables, constraint_nllq):

    return NonLinearLeastSquares(constraint_nllq, *variables)


class TestNonlinearLeastSquares:
    def test_localize(self, nllq):
        nllq.get_localized_object()

import numpy as np
import pytest


class TestNonLinearLeastSquares:
    def test_localize(self, variables, f_nlsq, constraint_nllq):
        variables[0].current_values = np.array([1., 2.])
        loc = constraint_nllq.get_localized_object()
        assert np.allclose(loc.residual(np.array([1., 2.])), np.array([0.,
                                                                       2.]))

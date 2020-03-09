import pytest


class TestNonlinearLeastSquares:
    def test_localize(self, nllq):
        nllq.get_localized_object()

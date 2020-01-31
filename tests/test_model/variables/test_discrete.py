import pytest
from pymloc.model.variables import Parameters
from pymloc.model.domains import RNDomain


class TestParameters:
    def test_domain(self):
        Parameters(4, domain=RNDomain(4))

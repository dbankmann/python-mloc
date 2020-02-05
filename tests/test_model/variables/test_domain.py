from pymloc.model.domains import RNDomain
from pymloc.model.variables import Parameters
import pytest


class TestRNDomain:
    def test_init_and_assertion(self):
        domain = RNDomain(4)
        params = Parameters(4, domain)
        with pytest.raises(TypeError):
            domain.in_domain(params)

    def test_in_domain_random(self):
        domain = RNDomain(4)
        params = Parameters(4, domain)
        params.current_values = params.get_random_values()
        assert domain.in_domain(params)

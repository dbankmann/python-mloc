from pymloc.model.domains import RNDomain
from pymloc.model.variables import Parameters


class TestRNDomain:
    def test_init_and_assertion(self):
        domain = RNDomain(4)
        params = Parameters(4, domain)
        domain.in_domain(params)

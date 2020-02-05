import pytest

from pymloc.model.domains import RNDomain
from pymloc.model.optimization.constraints.constraint import Constraint
from pymloc.model.optimization.local_optimization import LocalConstraint
from pymloc.model.optimization.local_optimization import LocalNullOptimization
from pymloc.model.optimization.local_optimization import LocalObjective
from pymloc.model.optimization.objectives.objective import Objective
from pymloc.model.optimization.optimization import NullOptimization
from pymloc.model.variables import InputStateVariables
from pymloc.model.variables import NullVariables
from pymloc.model.variables import ParameterContainer
from pymloc.model.variables.container import InputOutputStateVariables


@pytest.fixture
def variables():
    domain = RNDomain(1)
    parameters = ParameterContainer(1, domain)
    state_input = InputStateVariables(n_states=2, m_inputs=1)
    null_vars = NullVariables()
    return null_vars, parameters, state_input


@pytest.fixture
def local_opt():
    variables = InputOutputStateVariables(2, 4, 5)
    constraint = LocalConstraint()
    objective = LocalObjective()
    return LocalNullOptimization(objective, constraint,
                                 variables), variables, constraint, objective


@pytest.fixture
def opt():
    variables = InputOutputStateVariables(2, 4, 5)
    constraint = Constraint(*3 * (variables, ))
    objective = Objective(*3 * (variables, ))
    return NullOptimization(objective, constraint, variables, variables,
                            variables), variables, constraint, objective


def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item


def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail("previous test failed ({})".format(
                previousfailed.name))

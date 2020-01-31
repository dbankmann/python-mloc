from .optimization import OptimizationObject
from .constraints import Constraint
from .objectives import Objective
from ..variables import NullVariables
from ..variables import ParameterContainer, InputStateVariables
import numpy as np
import scipy.integrate


class PDOCObjective(Objective):
    def __init__(self, ll_vars, hl_vars, loc_vars, q, s, r):
        super().__init__(ll_vars, hl_vars, loc_vars)

        self._q = q
        self._s = s
        self._r = r
        self._test_matrices()

    def _test_matrices(self):
        m = self.local_level_variables.inputs.dimension
        n = self.local_level_variables.states.dimension
        np = self.higher_level_variables.parameters.dimension
        p, = self.higher_level_variables.get_random_values()
        x, u, t = self.local_level_variables.get_random_values()

        self._test_matrix_size(self._q(p, x, t), (n, n))
        self._test_matrix_size(self._s(p, x, t), (n, m))
        self._test_matrix_size(self._r(p, x, t), (m, m))

    def _test_matrix_size(self, matrix, shape):
        if matrix.ndim != 2 or matrix.shape != shape:
            raise ValueError("Matrix {} needs to have shape {}".format(
                matrix, shape))

    def _eval_weights(self, p, x, t):
        q = self._q(p, x, t)
        s = self._s(p, x, t)
        r = self._r(p, x, t)
        weights = np.block([[q, s], [s.T.conj(), r]])
        return weights

    def value(self, p, x, t):
        scipy.integrate()


class PDOCConstraint(Constraint):
    pass


class ParameterDependentOptimalControl(OptimizationObject):
    def __init__(self, parameters: ParameterContainer,
                 state_input: InputStateVariables, objective: PDOCObjective,
                 constraint: PDOCConstraint):
        if not isinstance(parameters, ParameterContainer):
            raise TypeError(parameters)
        self._parameters = parameters
        if not isinstance(state_input, InputStateVariables):
            raise TypeError(parameters)
        self._state_input = state_input
        lower_level_variables = NullVariables()
        local_level_variables = state_input
        higher_level_variables = parameters
        self._constraint = constraint
        self._objective = objective
        if not isinstance(objective, PDOCObjective):
            raise TypeError(objective)
        if not isinstance(constraint, PDOCConstraint):
            raise TypeError(constraint)
        super().__init__(self.objective, self.constraint,
                         lower_level_variables, higher_level_variables,
                         local_level_variables)

    @property
    def parameters(self):
        return self._parameters

    @property
    def state_input(self):
        return self._state_input

    @property
    def objective(self):
        return self._objective

    @property
    def constraint(self):
        return self._constraint

from .solver.linear.quadratic import QuadraticLinearSolver
from .solver.nonlinear.gauss_newton import GaussNewton
import numpy as np

"""
Definition of multilevel optimization problems as in thesis.
We assume that the constraint function G_i are equal for all i.
"""


class MultiLevelOptimalControl(object):
    """Driver routine that initializes and handles all the solvers.

    Currently assumes a buttom-to-top iterative structure.
    """

    model2standard_solvers = {'QuadraticOptimizationProblem': QuadraticLinearSolver,
                              'NonLinearLeastSquares': GaussNewton}

    def __init__(self, optimizations, solvers):
        self.optimizations = optimizations
        self.levels = len(self.optimizations)
        assert self.levels > 0
        self.solvers = solvers
        assert len(solvers) == self.levels



    def _get_solver_models(self):
        solvers_models = []
        for i, ocp in enumerate(self.optimizations):
            solvers_models.append({'model': ocp,
                            'solver': self.model2standard_solvers[ocp.__class__.__name__],
                            'level': i})
        return solvers_models


    def _initialize_solvers(self):
        for solver_model in self.solver_models:
            solver_model['solver_instance'] = solver_model['solver'](model=solver_model['model'])
        # TODO ...
        initial_state = np.array([[]])
        return initial_state

    def __call__(self):
        state = self._initialize_solvers()
        for i in range(4):
            for solver in self.solver_instances:
                state = solver.solve

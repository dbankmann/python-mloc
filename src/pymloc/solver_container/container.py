from collections import namedtuple


class SolverContainer:
    def __init__(self, problem, solver, default, creator_function):
        self._problem = problem
        self._solvers = []
        self._solver_tuple_obj = namedtuple('SolverTuple',
                                            'solver creator_function')
        self.add_solver(solver, default, creator_function)

    def add_solver(self, solver, default=False, creator_function=None):
        solver_tuple = self._solver_tuple_obj(solver, creator_function)
        self._solvers.append(solver_tuple)
        if default or not hasattr(self, "_default_solver"):
            self.default_solver = solver_tuple

    @property
    def problem(self):
        return self._problem

    @property
    def solvers(self):
        return self._solvers

    @property
    def default_solver(self):
        return self._default_solver

    @default_solver.setter
    def default_solver(self, solver_tuple):
        if solver_tuple not in self.solvers:
            raise ValueError("{} object should be a registered solver".format(
                solver_tuple.solver))
        self._default_solver = solver_tuple

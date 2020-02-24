class SolverContainer:
    def __init__(self, problem, solver, default):
        self._problem = problem
        self._solvers = []
        self.add_solver(solver, default)

    def add_solver(self, solver, default=False):
        self._solvers.append(solver)
        if default or not hasattr(self, "_default_solver"):
            self.default_solver = solver

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
    def default_solver(self, solver):
        if solver not in self.solvers:
            raise ValueError(
                "{} object should be a registered solver".format(solver))
        self._default_solver = solver

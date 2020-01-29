import inspect


class SolverContainer:
    def __init__(self, problem, solver, default):
        self._problem = problem
        self._solvers = []
        self.add_solver(solver, default)

    def add_solver(self, solver, default=False):
        self._solvers.append(solver)
        if default:
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
        if not solver in self.solvers:
            raise ValueError(
                "{} object should be a registered solver".format(solver))
        self._default_solver = solver


class SolverContainerFactory:
    __instance = None

    @staticmethod
    def get_instance():
        if SolverContainerFactory.__instance == None:
            SolverContainerFactory()
        return SolverContainerFactory.__instance

    def __init__(self):
        if SolverContainerFactory.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            SolverContainerFactory.__instance = self
        self._solvers = dict()

    def _check_problem(self, problem):
        if not inspect.isclass(problem):
            raise TypeError(problem)

    def register_solver(self, problem, solver, default=False):
        self._check_problem(problem)
        solver_container = self._solvers.get(problem)
        if solver_container is None:
            self._solvers[problem] = SolverContainer(problem, solver, default)
        else:
            solver_container.add_solver(solver, default)
        #Also register solvers for all problem subclasses.

        for subclass in problem.__subclasses__():
            self.register_solver(subclass, solver)

    def get_solver_container(self, problem):
        self._check_problem(problem)
        solver_container = self._solvers.get(problem)
        if solver_container is None:
            raise ValueError(
                "No registered solvers for problem: {}".format(problem))
        return solver_container


solver_container_factory = SolverContainerFactory()

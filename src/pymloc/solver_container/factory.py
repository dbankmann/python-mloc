from ..model import solvable
from . import SolverContainer
import inspect


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

    def _check_problem_instance(self, problem_instance):
        if not isinstance(problem_instance, solvable.Solvable):
            raise TypeError(problem_instance)

    def _get_problem_class(self, problem_instance):
        return problem_instance._class

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

    def get_solver_container(self, problem_instance):
        self._check_problem_instance(problem_instance)
        problem = self._get_problem_class(problem_instance)
        solver_container = self._solvers.get(problem)
        if solver_container is None:
            raise ValueError(
                "No registered solvers for problem: {}".format(problem))
        return solver_container


solver_container_factory = SolverContainerFactory.get_instance()

from ..solvers.solvers import SolverContainer
from abc import ABC
import inspect


class Solvable(ABC):
    _auto_generated = False

    def __init__(self):
        self._available_solvers = solver_container_factory.get_solver_container(
            self)
        self.set_default_solver()
        self._solver_instance = None

    def set_default_solver(self):
        self._solver = self._available_solvers.default_solver

    def solve(self):
        if not self._solver_instance:
            raise ValueError("Solver must be initialized first!")
        self._solver_instance.run()

    def init_solver(self, *args, **kwargs):
        self._solver_instance = self._solver(self, *args, **kwargs)

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
        if not isinstance(problem_instance, Solvable):
            raise TypeError(problem_instance)

    def _get_problem_class(self, problem_instance):
        try:
            return problem_instance._class
        except AttributeError:
            import ipdb; ipdb.set_trace()




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


solver_container_factory = SolverContainerFactory()

from ..solver_container import solver_container_factory
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

from ..solver_container import solver_container_factory
from abc import ABC
import inspect
import logging
logger = logging.getLogger()


class Solvable(ABC):
    _auto_generated = False

    def __init__(self):
        self.__get_class()
        self._available_solvers = solver_container_factory.get_solver_container(
            self)
        self.set_default_solver()
        self._solver_instance = None

    def set_default_solver(self):
        if self._available_solvers is None:
            logger.error("No solvers available. Cannot set default solver.")
        else:
            self._solver = self._available_solvers.default_solver

    def solve(self):
        try:
            self._solver_instance.run()
        except AttributeError:
            raise AttributeError(
                "There is currently no instantiated solver object for problem {}"
                .format(self))

    def init_solver(self, *args, **kwargs):
        self._solver_instance = self._solver(self, *args, **kwargs)

    def __get_class(self):
        if self._auto_generated:
            self._class = self._global_object._local_object_class
        else:
            self._class = self.__class__

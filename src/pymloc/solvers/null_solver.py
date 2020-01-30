from ..solver_container import solver_container_factory
from ..model.optimization import local_optimization
from . import BaseSolver


class NullSolver(BaseSolver):
    def run(self, *args, **kwargs):
        pass


solver_container_factory.register_solver(
    local_optimization.LocalNullOptimization, NullSolver, default=True)

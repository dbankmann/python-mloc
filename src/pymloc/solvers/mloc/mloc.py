from ...mloc import MultiLevelOptimalControl
from ...solver_container import solver_container_factory
from .. import BaseSolver


class MultiLevelIterativeSolver(BaseSolver):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self._opts = model.optimizations
        self._hopt = model.highest_opt
        self._lopt = model.lowest_opt
        self._vars = model.variables

    def _run(self, *args, **kwargs):
        self._check_variables()
        self._set_associations()
        self._init_solvers(*args, **kwargs)
        upperopt = self._hopt
        sol = upperopt.solve(*args, **kwargs)
        return sol

    def _check_variables(self):
        for variable in self._vars:
            if variable.current_values is None:
                raise ValueError(
                    "All Variables need to be initialized and the current value of {} is None."
                    .format(variable))

    def _set_associations(self):
        for variable, optimization in zip(self._vars, self._opts):
            variable.associated_problem = optimization.get_localized_object()
            variable.sensitivity_problem = optimization

    def _init_upper_level(self):
        self._hopt.init_solver()

    def _init_solvers(self):
        for opt in self._opts:
            opt.get_localized_object().init_solver(abs_tol=self.abs_tol,
                                                   rel_tol=self.rel_tol)


solver_container_factory.register_solver(MultiLevelOptimalControl,
                                         MultiLevelIterativeSolver,
                                         default=True)

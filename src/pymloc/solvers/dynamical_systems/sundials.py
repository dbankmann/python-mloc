from ..base_solver import BaseSolver


class DAESundials(BaseSolver):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

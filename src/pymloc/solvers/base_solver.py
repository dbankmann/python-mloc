from abc import ABC, abstractmethod


class BaseSolver(ABC):
    def __init__(self, model, abs_tol=1.e-3, rel_tol=1.e-3, max_iter=10):
        self.abs_tol = abs_tol
        self.model = model
        self.rel_tol = rel_tol
        self.max_iter = max_iter

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def output(self):
        raise NotImplementedError

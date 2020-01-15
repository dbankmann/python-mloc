class OptimizationProblem(object):
    """
    Class for defining optimization problems.


    Parameters
    ----------
    np: number of (upper level) parameters
    nn: number of states
    nm: number of inputs

    """
    def __init__(self, nn: int, nm: int, np: int):
        pass

    def residual(self):
        raise NotImplementedError



class NullOptimization(OptimizationProblem):
    def __init__(self):
        super().__init__(nn=0, nm=0, np=0)

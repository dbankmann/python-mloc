from typing import Callable

import numpy as np

TimeCallable = Callable[[float], np.ndarray]
ParameterCallable = Callable[[np.ndarray], np.ndarray]
ParameterTimeCallable = Callable[[np.ndarray, float], np.ndarray]

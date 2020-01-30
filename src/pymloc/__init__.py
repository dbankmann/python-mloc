__version__ = '0.0.1'

# Models have to be imported before solvers, because of the coupling between them in solver_container_factory
from . import model
from . import solver_container
from . import solvers

from .model import variables as variables
from .model import optimization as optimization

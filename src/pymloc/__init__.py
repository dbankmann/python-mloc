__version__ = '0.0.1'

#Enable 64 bit on startup
import jax.config

# Models have to be imported before solvers, because of the coupling between them in solver_container_factory
from . import logger
from . import model
from . import solver_container
from . import solvers
from .mloc import MultiLevelOptimalControl

jax.config.update("jax_enable_x64", True)

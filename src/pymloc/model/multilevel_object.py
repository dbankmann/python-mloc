import inspect
import logging
from abc import ABC
from abc import abstractmethod

from .variables.container import VariablesContainer

logger = logging.getLogger(__name__)


class MultiLevelObject(ABC):
    def __init__(self, lower_level_variables: VariablesContainer,
                 higher_level_variables: VariablesContainer,
                 local_level_variables: VariablesContainer):
        self._higher_level_variables = higher_level_variables
        self._lower_level_variables = lower_level_variables
        self._local_level_variables = local_level_variables
        for container in lower_level_variables, higher_level_variables, local_level_variables:
            if not isinstance(container, VariablesContainer):
                raise TypeError(container)

        self._localize_dict = dict()

    @property
    def lower_level_variables(self):
        return self._lower_level_variables

    @property
    def higher_level_variables(self):
        return self._higher_level_variables

    @property
    def local_level_variables(self):
        return self._local_level_variables

    def _hl_vars_filter(self):
        return self._higher_level_variables.current_values

    def _ll_vars_filter(self):
        return self._lower_level_variables.current_values

    def _loc_vars_filter(self):
        return self._local_level_variables.current_values

    def _get_ll_vars(self, loc_vars):
        id_vars = id(loc_vars)
        ll_vars = self._localize_dict.get(id_vars)
        if ll_vars is None:
            logger.info("Updating lower level variables...")
            self._lower_level_variables.update_values()
            ll_vars = self._ll_vars_filter()
            self._localize_dict[id_vars] = ll_vars
        return ll_vars

    def localize_method(self, method):
        if method is None:
            return None
        nparam = len(inspect.signature(method).parameters)
        #Signature 'guessing': For 2 parameters, only use hl_vars and loc_vars
        #TODO: Generalize to more cases
        if nparam == 3:

            def localized_function(variables):
                hl_vars = self._hl_vars_filter()
                ll_vars = self._get_ll_vars(variables)
                args = (ll_vars, ) + (hl_vars, ) + (variables, )
                return method(*args)

            return localized_function
        elif nparam == 2:

            def localized_function(variables):
                hl_vars = self._hl_vars_filter()
                args = (hl_vars, ) + (variables, )
                return method(*args)

            return localized_function

        elif nparam == 1:
            hl_vars = self._hl_vars_filter()
            localized_parameter = method(hl_vars)
            return localized_parameter
        elif nparam == 0:
            logger.warning(
                "Localization of method without arguments requested.")
            return method
        else:
            logger.error("Unsupported number of parameters")
            return None

    def get_localized_object(self, **kwargs):
        return local_object_factory.get_localized_object(self, **kwargs)

    def solve(self, *args, **kwargs):
        loc_object = self.get_localized_object()
        loc_object.init_solver()
        solution = loc_object.solve()
        return solution


class LocalObjectFactory:
    def __init__(self):
        self._localizer_objects = {}

    def register_localizer(self, global_object, localizer_object):
        if not issubclass(global_object, MultiLevelObject):
            raise ValueError(global_object)
        self._localizer_objects[global_object] = localizer_object

    def get_localized_object(self, global_object, *args, **kwargs):
        localizer_object = self._localizer_objects.get(global_object.__class__)
        if not localizer_object:
            raise ValueError(
                "No localizer found for global object class: {}".format(
                    global_object))
        return localizer_object(global_object, **kwargs)


local_object_factory = LocalObjectFactory()

#
# Copyright (c) 2019-2020
#
# @author: Daniel Bankmann
# @company: Technische Universität Berlin
#
# This file is part of the python package pymloc
# (see https://gitlab.tubit.tu-berlin.de/bankmann91/python-mloc )
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#
import inspect
import logging
from abc import ABC
from typing import Callable
from typing import Tuple
from typing import Type
from typing import Union

from . import Solvable
from .variables.container import VariablesContainer

logger = logging.getLogger(__name__)


class MultiLevelObject(ABC):
    """
    BaseObject for all classes that describe problems, with multiple variable
    levels.
    These can be both, higher level variables and lower level variables.
    """
    _local_object_class: Type[Solvable]

    def __init__(self, lower_level_variables: VariablesContainer,
                 higher_level_variables: VariablesContainer,
                 local_level_variables: VariablesContainer):
        self._higher_level_variables = higher_level_variables
        self._lower_level_variables = lower_level_variables
        self._local_level_variables = local_level_variables
        for container in lower_level_variables, higher_level_variables, local_level_variables:
            if not isinstance(container, VariablesContainer):
                raise TypeError(container)

        self._localize_id: Union[int, None] = None
        self._localize_val: Union[float, None] = None
        self.ll_sens_selector: Union[Callable, None] = None
        self.ll_sens_selector_shape: Union[Tuple[int], None] = None

    @property
    def local_object_class(self):
        return self._local_object_class

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

    def _get_ll_solver_args(self):
        return dict()

    def _get_ll_vars(self, loc_vars):
        id_vars = id(loc_vars)
        if self._localize_id is None or id_vars != self._localize_id:
            logger.info("Updating lower level variables...")
            kwargs = self._get_ll_solver_args()
            self._lower_level_variables.update_values(**kwargs)
            ll_vars = self._ll_vars_filter()
            self._localize_value = ll_vars
        return self._localize_value

    def localize_method(self, method):
        if method is None:
            return None
        logger.debug("Getting localized method for method: {}".format(method))
        nparam = len(inspect.signature(method).parameters)
        # Signature 'guessing': For 2 parameters, only use hl_vars and loc_vars
        # TODO: Generalize to more cases
        if nparam == 3:

            def localized_function(variables):
                hl_vars = self._hl_vars_filter()
                ll_vars = self._get_ll_vars(variables)
                args = (ll_vars, ) + (hl_vars, ) + (variables, )
                logger.debug("In localized method: {}\nargs: {}".format(
                    method, args))
                return method(*args)

            return localized_function
        elif nparam == 2:

            def localized_function(variables):
                hl_vars = self._hl_vars_filter()
                args = (hl_vars, ) + (variables, )
                logger.debug("In localized method: {}\nargs: {}".format(
                    method, args))
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
        loc_var = self.local_level_variables
        solution = loc_var.update_values()
        return solution


class LocalObjectFactory:
    "Class for maintaining mappings between global multilevel and localized objects."

    def __init__(self):
        self._localizer_objects = {}

    def register_localizer(self, global_object: Type[MultiLevelObject],
                           localizer_object):
        if not issubclass(global_object, MultiLevelObject):
            raise ValueError(global_object)
        self._localizer_objects[global_object] = localizer_object

    def get_localized_object(self, global_object: Type[MultiLevelObject],
                             *args, **kwargs):
        localizer_object = self._localizer_objects.get(global_object.__class__)
        if not localizer_object:
            raise ValueError(
                "No localizer found for global object class: {}".format(
                    global_object))
        return localizer_object(global_object, **kwargs)


local_object_factory = LocalObjectFactory()

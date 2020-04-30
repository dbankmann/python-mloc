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
from abc import ABC
from abc import abstractmethod


class LocalObjective(ABC):
    def __init__(self):
        pass


class LocalNullObjective(LocalObjective):
    def value(self):
        pass

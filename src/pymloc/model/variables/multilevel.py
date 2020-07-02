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


# TODO: Check if can be removed
class MultiLevelVariables(object):
    def __init__(self, variables_list=None):
        if variables_list is not None:
            for variable in variables_list:
                self.add_variable_level(variable)

    def add_variable_level(variable, level=None):
        pass

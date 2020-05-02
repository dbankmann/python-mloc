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
import logging.config

import numpy as np
import yaml
from pkg_resources import resource_stream


def load_config():
    _log_config_file = 'config.yaml'
    _log_config_location = resource_stream(
        __name__.rpartition('.')[0] + ".config", _log_config_file)
    config = yaml.safe_load(_log_config_location.read())

    return config


def prepare_logger(logging_config):
    logging.config.dictConfig(logging_config)
    np.set_printoptions(linewidth=200)


logging_config = load_config()
prepare_logger(logging_config)

import logging

from .solvers.base_solver import solver_level


class LevelFilter(logging.Filter):
    def filter(self, record):
        level = solver_level.level

        indents = level * '\t'
        record.msg = record.msg.replace('\n', '\n' + indents)
        record.level_indent = indents
        return level < 4


rootlogger = logging.getLogger()
rootlogger.handlers[0].addFilter(LevelFilter())

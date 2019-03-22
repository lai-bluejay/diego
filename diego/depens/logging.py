#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
depens/logging.py was created on 2019/03/21.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""

from __future__ import absolute_import

import colorlog
import logging
from logging import CRITICAL  
from logging import DEBUG  
from logging import ERROR  
from logging import FATAL  
from logging import FATAL  
from logging import INFO  
from logging import WARN  
from logging import WARNING  
import threading

_lock = threading.Lock()
_default_handler = None  # type: Optional[logging.Handler]


def create_default_formatter():
    # type: () -> colorlog.ColoredFormatter
    """Create a default formatter of log messages.

    This function is not supposed to be directly accessed by library users.
    """

    return colorlog.ColoredFormatter(
        '%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s')


def _get_library_name():
    # type: () -> str

    return __name__.split('.')[0]


def _get_library_root_logger():
    # type: () -> logging.Logger

    return logging.getLogger(_get_library_name())


def _configure_library_root_logger():
    # type: () -> None

    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()
        _default_handler.setFormatter(create_default_formatter())

        python_root_logger = logging.getLogger()
        if python_root_logger.handlers:
            # Users have already configured python root logger. This library's log outputs will be
            # propagated to the root logger, and thus they will be collected properly. We don't
            # further configure loggers by ourselves to prevent double logging, etc.
            return

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(logging.INFO)


def _reset_library_root_logger():
    # type: () -> None

    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None


def get_logger(name):
    # type: (str) -> logging.Logger
    """Return a logger with the specified name.

    This function is not supposed to be directly accessed by library users.
    """

    _configure_library_root_logger()
    return logging.getLogger(name)


def get_verbosity():
    # type: () -> int
    """Return the current level for the diego's root logger.

    Returns:
        Logging level, e.g., ``diego.logging.DEBUG`` and ``diego.logging.INFO``.

    .. note::
        diego has following logging levels:

        - ``diego.logging.CRITICAL``, ``diego.logging.FATAL``
        - ``diego.logging.ERROR``
        - ``diego.logging.WARNING``, ``diego.logging.WARN``
        - ``diego.logging.INFO``
        - ``diego.logging.DEBUG``
    """

    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity):
    # type: (int) -> None
    """Set the level for the diego's root logger.

    Args:
        verbosity:
            Logging level, e.g., ``diego.logging.DEBUG`` and ``diego.logging.INFO``.
    """

    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


def disable_default_handler():
    # type: () -> None
    """Disable the default handler of the diego's root logger.

    Example:

        Stop and then resume logging to standard output.

        .. code::

            >> study = diego.create_study()
            >> diego.logging.disable_default_handler()
            >> study.optimize(objective, n_trials=10)
            >> len(study.trials)
            10
            >> diego.logging.enable_default_handler()
            >> study.optimize(objective, n_trials=10)
            [I 2018-11-07 16:11:28,285] Finished a trial resulted in value: 3787.44371584515. ...
    """

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)


def enable_default_handler():
    # type: () -> None
    """Enable the default handler of the diego's root logger.

    Please refer to the example shown in :func:`~diego.logging.disable_default_handler()`.
    """

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)

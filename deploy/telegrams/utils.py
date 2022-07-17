#!/usr/bin/env python
# -*- coding: utf-8 -*-

from importlib import import_module
from logzero import logger


def telegram_cmd(tag):
    # デコレータ
    def _telegram_cmd(func):
        def wrapper(*args, **kwargs):
            logger.info("\033[34m" + f"|--> {tag}" + "\033[0m")
            return func(*args, **kwargs)
        return wrapper
    return _telegram_cmd


def get_module(path, **kwargs):
    path, module_name = str(path).rsplit(".", 1)
    module_cls = getattr(import_module(path), module_name)
    return module_cls(**kwargs)

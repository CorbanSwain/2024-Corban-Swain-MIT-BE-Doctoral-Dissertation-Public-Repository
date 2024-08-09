#!python3
# meta.py

# Corban Swain, 2021

import functools as ft
import inspect

__all__ = ['get_class_that_defined_method',
           'get_full_func_name']

from typing import Callable


def get_class_that_defined_method(meth: Callable):
    # https://stackoverflow.com/a/25959545

    if isinstance(meth, ft.partial):
        return get_class_that_defined_method(meth.func)

    if (inspect.ismethod(meth)
            or (inspect.isbuiltin(meth)
                and getattr(meth, '__self__', None) is not None
                and getattr(meth.__self__, '__class__', None))):

        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls

        # fallback to __qualname__ parsing
        meth = getattr(meth, '__func__', meth)

    if inspect.isfunction(meth):
        cls = getattr(
            inspect.getmodule(meth),
            meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
            None)
        if isinstance(cls, type):
            return cls

    # handle special descriptor objects
    return getattr(meth, '__objclass__', None)


def get_full_func_name(func: Callable) -> str:
    cls = get_class_that_defined_method(func)
    if cls:
        cls_str = str(cls).split("'")[1]
    else:
        cls_str = None

    if cls_str:
        return '.'.join([cls_str, func.__name__])
    else:
        return func.__name__

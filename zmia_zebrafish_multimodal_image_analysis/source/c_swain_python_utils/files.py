#!python3
# files.py

from __future__ import annotations
import os
import warnings

__all__ = ['get_filepath', 'touchdir', 'make_str_filesafe',
           'no_ext_basename', 'get_file_dir']


def no_ext_basename(f: bytes | str | os.PathLike,
                    **kwargs) -> str:
    if kwargs:
        warnings.warn('`no_ext_basename` accepts only one argument; use of the '
                      '`extension_sep` parameter is deprecated.')
    return os.path.splitext(os.path.basename(f))[0]


def get_filepath(fle):
    """deprecated, do not use"""
    warnings.warn('This function `get_filepath` is deprecated; do not use.')
    return os.path.dirname(os.path.abspath(fle))


def touchdir(pth, recur=True):
    try:
        os.mkdir(pth)
    except FileNotFoundError:
        if recur:
            head, tail = os.path.split(pth)
            if head:
                touchdir(head)
                touchdir(pth)
        else:
            raise
    except FileExistsError:
        pass


def make_str_filesafe(string):
    filesafe_replacements = [
        ('\"', '[`]'),
        ('\\', '[~]'),
        ('/', '[~~]'),
        ('?', '[!!]'),
        ('<', '[^]'),
        ('>', '[v]'),
        ('*', '[&]'),
        (':', '[#]'),
        (' ', '_'), ]
    for a, b in filesafe_replacements:
        string = string.replace(a, b)
    return string


def get_file_dir(file_path):
    return os.path.realpath(os.path.dirname(file_path))

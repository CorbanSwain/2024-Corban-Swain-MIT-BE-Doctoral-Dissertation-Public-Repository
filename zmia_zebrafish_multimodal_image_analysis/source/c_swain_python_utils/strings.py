#!python3
# strings.py

__all__ = ['nowstr']

import datetime


def nowstr():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M')
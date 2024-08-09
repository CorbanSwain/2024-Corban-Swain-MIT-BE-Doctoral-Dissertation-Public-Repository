#!python3
# numpy.py

# Corban Swain, 2021

__all__ = ['zero_center_range']


def zero_center_range(length):
    from numpy import arange

    return arange(length) - ((length - 1) / 2)

#!python3
# timing.py

# Project: c_swain_python_utils
# by Corban Swain 2019

import timeit
import time
import datetime as dt
import functools

__all__ = ['time_func', 'timed', 'log_time_delta']


def time_func(func, n=1000, logger=None):
    # TODO - implement log level selection
    log = logger.info if logger else print
    timeit.timeit()
    t = timeit.Timer(func)
    n_seconds = t.timeit(int(n))

    log('Time elapsed for {:e} repeats: {:s}'
        .format(n, format_seconds(n_seconds)))


def timed(logger=None):
    # TODO - implement log level selection
    log = logger.info if logger else print

    def timed_decorator(func):
        @functools.wraps(func)
        def func_wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            log('{:15s} took {:s}.'
                .format(func.__name__, format_seconds(end - start)))
            return result
        return func_wrapper
    return timed_decorator


def log_time_delta(logger, start, msg):
    # TODO - implement log level selection
    logger.debug('{:s} took {:s}'.format(msg, format_seconds(time.time()-start)))


def format_seconds(seconds):
    # TODO - add format specification
    time_delta = dt.timedelta(seconds=seconds)
    return '{} h:m:s'.format(time_delta)

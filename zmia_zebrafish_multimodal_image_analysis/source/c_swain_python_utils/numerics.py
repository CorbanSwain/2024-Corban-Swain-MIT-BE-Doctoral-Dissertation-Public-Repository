#!/bin/python3
# numerics.py

__all__ = ['logical_or', 'logical_and']


def logical_or(*args):
    from numpy import logical_or

    return binary_ndarray_reduce(*args,
                                 binary_fcn=logical_or,
                                 identity=False)


def logical_and(*args):
    from numpy import logical_and

    return binary_ndarray_reduce(*args,
                                 binary_fcn=logical_and,
                                 identity=True)


def binary_ndarray_reduce(*args, binary_fcn, identity):
    if len(args) == 0:
        return None
    elif len(args) == 1:
        return binary_ndarray_reduce(identity, args[0],
                                     binary_fcn=binary_fcn,
                                     identity=identity)
    elif len(args) == 2:
        return binary_fcn(*args)
    else:
        return binary_ndarray_reduce(
            args[0],
            binary_ndarray_reduce(*args[1:],
                                  binary_fcn=binary_fcn,
                                  identity=identity),
            binary_fcn=binary_fcn,
            identity=identity
        )

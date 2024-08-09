#!python3
# dicts.py

# Project: c_swain_python_utils
# by Corban Swain 2019

import collections.abc

__all__ = ['sort_dict', 'sort_mapping']


def sort_dict(d: dict, *, recursive: bool = False, key=repr) -> dict:
    return sort_mapping(d,
                        recursive=recursive,
                        key=key,
                        recur_on_class=dict)


def sort_mapping(d,
                 *,
                 recursive: bool = False,
                 key=repr,
                 recur_on_class=collections.abc.Mapping):

    output_d = d.__class__()
    sorted_keys = sorted(d.keys(), key=key)

    for k in sorted_keys:
        if recursive and isinstance(d[k], recur_on_class):
            output_d[k] = sort_mapping(d[k],
                                       recursive=True,
                                       key=key,
                                       recur_on_class=recur_on_class)
        else:
            output_d[k] = d[k]

    return output_d


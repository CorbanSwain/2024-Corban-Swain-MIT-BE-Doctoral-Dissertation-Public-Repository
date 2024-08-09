#!python3
# prairie_view_imports.py

# shared utilities
from __future__ import annotations
from utilities import *

import os
import pickle
import glob
import datetime as dt
import numpy as np
import skimage as ski
import skimage.io
import itertools as it
import functools as ft
import napari
import copy
import matplotlib.pyplot as plt
import bottleneck as bn
import argparse
import logging
from multiprocessing.pool import Pool
import abc
from collections import UserDict, UserList
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass
import operator
import enum
import c_swain_python_utils as csutils

# PV Metadata Constants


class PVMetadataStandardKeys:
    # metadata keys for parsing Prairie View acquisition .xml files
    PV_STATE_SHARD = 'PVStateShard'
    PV_STATE_VALUE = 'PVStateValue'
    SUBINDEXED_VALUES = 'SubindexedValues'
    SUBINDEXED_VALUE = 'SubindexedValue'
    INDEXED_VALUE = 'IndexedValue'
    SEQUENCE = 'Sequence'
    FRAME = 'Frame'
    EXTRA_PARAMETERS = 'ExtraParameters'
    FILE = 'File'
    KEY = 'key'
    INDEX = 'index'
    SUBINDEX = 'subindex'
    VALUE = 'value'
    DESCRIPTION = 'description'
    CHANNEL_NUM = 'channel'
    RELATIVE_TIME = 'relativeTime'
    ABSOLUTE_TIME = 'absoluteTime'
    BIT_DEPTH = 'bitDepth'
    LINES_PER_FRAME = 'linesPerFrame'
    PIXELS_PER_LINE = 'pixelsPerLine'
    MICRONS_PER_PIXEL = 'micronsPerPixel'
    SEQUENCE_TYPE = 'type'
    TRACK = 'track'
    PARAMETER_SET = 'parameterSet'
    FILENAME = 'filename'
    CYCLE = 'cycle'
    CHANNEL_NAME = 'channelName'
    POSITION_CURRENT = 'positionCurrent'
    X_AXIS = 'XAxis'
    Y_AXIS = 'YAxis'
    Z_AXIS = 'ZAxis'
    FILE_PAGE = 'page'
    DATE = 'date'
    NOTES = 'notes'
    LASER_WAVELENGTH = 'laserWavelength'
    LASER_POWER = 'laserPower'
    LASER_ATTENUATION = 'laserPowerAttenuation'
    OBJECTIVE_LENS = 'objectiveLens'
    DWELL_TIME = 'dwellTime'
    TIME = 'time'
    BIDIRECTIONAL_Z_SCAN = 'bidirectionalZ'


class PVMetadataKeys(PVMetadataStandardKeys):
    # custom keys specific to these scripts
    CUSTOM_EXTRA_PARAMETERS = 'extraParameters'
    CHANNEL_INDEX = 'channelIndex'
    TRACK_INDEX = 'trackIndex'
    FRAME_INDEX = 'frameIndex'
    TRACK_ID = 'trackId'
    FRAME_ID = 'frameId'
    TWO_D_IMAGE_ID = 'singleImageId'
    SEQUENCE_INDEX = 'sequenceIndex'
    TIMESTAMP = 'timestamp'
    PRECISE_TIMESTAMP = 'preciseTimestamp'


# convenience renaming
pvmk = PVMetadataKeys


def get_index_correction(is_one_indexed: bool) -> int:
    return 1 if is_one_indexed else 0


PV_FRAMES_ONE_INDEXED:      bool = True
PV_CHANNELS_ONE_INDEXED:    bool = True
PV_TRACKS_ONE_INDEXED:      bool = True
PV_CYCLES_ONE_INDEXED:      bool = True
PV_PAGE_ONE_INDEXED:        bool = True
PV_FRAME_INDEX_CORRECTION:   int = get_index_correction(PV_FRAMES_ONE_INDEXED)
PV_CHANNEL_INDEX_CORRECTION: int = get_index_correction(PV_CHANNELS_ONE_INDEXED)
PV_TRACK_INDEX_CORRECTION:   int = get_index_correction(PV_TRACKS_ONE_INDEXED)
PV_CYCLE_INDEX_CORRECTION:   int = get_index_correction(PV_CYCLES_ONE_INDEXED)
PV_PAGE_INDEX_CORRECTION:    int = get_index_correction(PV_PAGE_ONE_INDEXED)

MAIN_LASER_POCKELS_INDEX = '0'

# General Constants
PV_DATETIME_FMT = '%m/%d/%Y %I:%M:%S %p'
PV_TIME_FMT = '%H:%M:%S.%f'
DATETIME_FMT = '%Y%d%m %H:%M:%S.%f'
DESCRIPTION_KEY_ADDON = '-description'
POSITION_UNITS = 'micrometer'
FLOAT_TIME_UNITS = 'second'

# Globals
log: logging.Logger = csutils.get_logger(__name__)
debug_mode = False

files = []


@dataclass
class Entry:
    """
    base class for a metadata entry. all Entry instances have an attribute
    `value` holding the Entry's value.
    """
    value: Any


@dataclass
class EmptyEntry(Entry):
    """
    Entry subclass representing an entry with no value
    """

    def __init__(self):
        super().__init__(None)


@dataclass
class DescriptiveEntry(Entry):
    """
    metadata entry with a description

    Example
    -------
    >>> x = dict()
    >>> x['position'] = DescriptiveEntry(135, 'z-piezo')
    """
    description: str


class Metadata(UserDict):
    """
    A data structure to represented nested dictionaries. Uses tuple keys that
    flatten indexing into nested dicts. This is a base class and should not be
    used directly.

    Example
    _______
    >>> x = Metadata.from_dict({'y': {'z': 42}})
    >>> print(x[('y', 'z')] == 42)
    True
    """
    # class attributes
    KEY_TYPES = (tuple,)

    # magic methods
    def __new__(cls, *args, **kwargs):
        if cls is Metadata:
            raise TypeError(f'{cls.__name__} class cannot be instantiated '
                            f'directly, it can only be used as a base class. ')
        return super().__new__(cls, *args, **kwargs)

    def __init__(self):
        # intentionally do not allow for initialization from key value entries
        self.lock: bool = False
        self._flat_key_set_cache: Optional[set[tuple[str, ...]]] = None
        super().__init__()

    def __setitem__(self,
                    key: str | tuple[str, ...],
                    value: Any):
        self._fail_on_lock()
        tup_key = self.key_to_tuple_key(key)
        self.data[tup_key] = value

    def __getitem__(self, key):
        try:
            return self.data[key]
        except KeyError:
            pass

        tup_key = self.key_to_tuple_key(key)
        try:
            return self.data[tup_key]
        except KeyError:
            # the following code allows the flat dict to be indexed like a
            #  nested dict
            key_len = len(tup_key)
            sub_md = self.__class__()
            for k in self.data.keys():
                if k[:key_len] == tup_key:
                    sub_md.data[k[key_len:]] = self.data[k]
            if sub_md:
                return sub_md
            else:
                raise

    def __delitem__(self, key):
        self._fail_on_lock()
        try:
            del self.data[key]
        except KeyError:
            pass
        else:
            return

        tup_key = self.key_to_tuple_key(key)
        try:
            del self.data[tup_key]
        except KeyError:
            # the following code allows the flat dict to have entries deleted
            #  like a nested dict
            key_len = len(tup_key)
            for k in self.data.keys():
                if k[:key_len] == tup_key:
                    del self.data[k]

    def __contains__(self, key):
        if key in self.data:
            return True
        tup_key = self.key_to_tuple_key(key)
        return tup_key in self.data

    def __repr__(self):
        return f'{self.__class__.__name__}(len={len(self.data):d})'

    # general methods
    def update_with_metadata(self, other):
        """
        modifies the metadata object in place given an updating metadata object
        """
        if debug_mode:
            updated_dict = self
        else:
            updated_dict = self.data

        for k, v in other.data.items():
            updated_dict[k] = v

    def shared_keys(self, other: Metadata) -> set:
        """
        returns the keys shared by another metadata object
        """
        return self.flat_key_set & other.flat_key_set

    def different_keys(self, other: Metadata) -> set:
        """
        returns the keys in this metadata object NOT found in another medadata
        object
        """
        return self.flat_key_set - other.flat_key_set

    def sorted(self):
        return csutils.sort_mapping(self,
                                    recursive=True,
                                    recur_on_class=Metadata)

    def clear_all_terminal_values(self, *, fill_value=None):
        """
        This method sets all non-metadata values in the dict to `fill_value`
        recursively. Useful for creating a template metadata object.
        """
        for k in self.flat_key_set:
            self[k] = EmptyEntry() if fill_value is None else fill_value

    def get_value(self, key):
        """
        will always return the value from an entry in the metadata; the raw
        value if that's what is stored and the `value` attribute if an
        Entry object is stored.
        """
        entry_like = self[key]
        return self._get_value_from_entry(self[key])

    def to_nested_dict(self):
        """
        converts the metadata object to a nested dictionary
        """
        out_dict = dict()
        for tup_k, v in self.data.items():
            sub_dict_keys = tup_k[:-1]
            value_key = tup_k[-1]
            value_dict = out_dict
            for sub_k in sub_dict_keys:
                try:
                    value_dict = value_dict[sub_k]
                except KeyError:
                    value_dict = value_dict[sub_k] = dict()
            value_dict[value_key] = v
        return out_dict

    # protected methods
    def _fail_on_lock(self):
        if self.lock:
            msg = 'Metadata object is locked; cannot execute command.'
            raise RuntimeError(msg)

    def _setitem_from_metadata(self, key, metadata):
        for k, v in metadata.data.items():
            new_key = self.key_to_tuple_key(key) + k
            self.data[new_key] = v

    # getters and setters
    @property
    def flat_key_set(self) -> set[tuple[str, ...]]:
        if self.lock and (self._flat_key_set_cache is not None):
            return self._flat_key_set_cache

        key_set = set(self.data.keys())

        if self.lock:
            self._flat_key_set_cache = key_set

        return key_set

    # class methods
    @classmethod
    def from_metadata(cls, init_md, /):
        new_metadata = cls()
        new_metadata.update_with_metadata(init_md)
        return new_metadata

    @classmethod
    def from_dict(cls, init_dict, /, *, check_for_tup_keys=False):
        new_metadata = cls()
        flat_dict = cls._flatten_dict(init_dict,
                                      check_for_tup_keys=check_for_tup_keys)
        if debug_mode:
            for k, v in flat_dict.items():
                new_metadata[k] = v
        else:
            new_metadata.data = flat_dict
        return new_metadata

    @classmethod
    def _flatten_dict(cls, d, /, *, check_for_tup_keys):
        output_dict = dict()
        for k, v in d.items():
            if check_for_tup_keys:
                tup_k = cls.key_to_tuple_key(k)
            else:
                tup_k = (k,)
            if isinstance(v, (dict, UserDict)):
                flat_dict = cls._flatten_dict(
                    v, check_for_tup_keys=check_for_tup_keys)
                for kk, vv in flat_dict.items():
                    new_key = tup_k + kk
                    output_dict[new_key] = vv
            else:
                output_dict[tup_k] = v
        return output_dict

    @classmethod
    def key_list_to_tuple_key_list(cls, key_list):
        return [cls.key_to_tuple_key(k) for k in key_list]

    # static methods

    @staticmethod
    def key_to_tuple_key(key: Any) -> tuple:
        return key if isinstance(key, tuple) else (key,)

    @staticmethod
    def _get_value_from_entry(entry: Entry | Any) -> Any:
        if isinstance(entry, Entry):
            return entry.value
        else:
            return entry


class ScalarMetadata(Metadata):
    """
    Subclass of Metadata in which all entries are scalar (non-list) values.
    """

    ENTRY_TYPES = (Metadata, Entry, int, float, str)

    # magic methods
    def __setitem__(self, key, value):
        self._fail_on_lock()
        if not isinstance(value, self.ENTRY_TYPES):
            raise TypeError(
                f'{ScalarMetadata.__name__} values must be of type '
                f'{tuple((t.__name__ for t in self.ENTRY_TYPES))}; got a value '
                f'of type {type(value).__name__} when setting key "{key}".')

        if isinstance(value, Metadata):
            if not isinstance(value, ScalarMetadata):
                raise TypeError(f'Nested {Metadata.__name__} values must be of '
                                f'type {ScalarMetadata.__name__}; '
                                f'got a value of type {type(value).__name__} '
                                f'when setting key "{key}".')
            self._setitem_from_metadata(key, value)

        else:
            super().__setitem__(key, value)

    # general methods
    def remove_templated_entries(self, template_metadata, *, ignore_list=()):
        self._fail_on_lock()
        ignore_list = self.key_list_to_tuple_key_list(ignore_list)

        for k in self.shared_keys(template_metadata):
            if not (ignore_list and k in ignore_list):
                del self[k]


class ArrayedMetadata(Metadata):
    """
    Subclass of metadata in which some entries can be list/arrayed and others
    can be scalar. Useful for representing metadata values which change
    across time/acquisition events.
    """

    ENTRY_TYPES = ScalarMetadata.ENTRY_TYPES + (list,)

    # magic methods
    def __setitem__(self, key, value):
        self._fail_on_lock()
        if not isinstance(value, self.ENTRY_TYPES):
            raise TypeError(
                f'{ArrayedMetadata.__name__} values must be of type '
                f'{tuple(t.__name__ for t in self.ENTRY_TYPES)}; got a '
                f'value of type {type(value).__name__} when setting key '
                f'"{key}".')

        if isinstance(value, Metadata):
            if not isinstance(value, ArrayedMetadata):
                raise TypeError(f'Nested {Metadata.__name__} values must be of '
                                f'type {ArrayedMetadata.__name__}; '
                                f'got a value of type {type(value).__name__} '
                                f'when setting key "{key}".')
            self._setitem_from_metadata(key, value)
        else:
            super().__setitem__(key, value)

    # general methods
    def update_with_metadata(self, other):
        if debug_mode:
            # this will make a call to the entry-checking __setitem__ method
            updated_dict = self
        else:
            updated_dict = self.data

        for k, v in other.data.items():
            updated_dict[k] = copy.deepcopy(v) if isinstance(v, list) else v

    def value_append_if_missing(self, other: ScalarMetadata):
        self._fail_on_lock()
        for k in self.shared_keys(other):
            if isinstance(self[k], list):
                self[k].append(other[k])
            # if self[k] is not a list then that value should remain a scalar
            # non-arrayed value, only values that were not originally in the
            # self should become arrayed

        for k in other.different_keys(self):
            self[k] = [other[k], ]

    def _get_value_from_entry(self, entry):
        if isinstance(entry, list):
            return [self._get_value_from_entry(ele)
                    for ele in entry]
        else:
            return super()._get_value_from_entry(entry)

    # class methods

    @classmethod
    def entry_nary_op(cls, op, *args):
        if any(isinstance(e, list) for e in args):
            entry_iterators = []
            for entry in args:
                if isinstance(entry, list):
                    entry_iterators.append(entry)
                else:
                    entry_iterators.append(it.repeat(entry))
            return [cls.entry_nary_op(op, *unpacked_args)
                    for unpacked_args in zip(*entry_iterators)]
        else:
            return op(*args)

    @classmethod
    def entry_binop(cls, op, a, b):
        """
        performs the binary operation op between the (possibly) arrayed entries
        `a` and `b`, returning a (possibly) arrayed entry list or a scalar entry
        """
        return cls.entry_nary_op(op, a, b)

    @classmethod
    def linearize_entry(cls, entry):
        if isinstance(entry, list):
            out_list = []
            for e in entry:
                if isinstance(e, list):
                    out_list.extend(cls.linearize_entry(e))
                else:
                    out_list.append(e)
            return out_list
        else:
            return [entry, ]

    @classmethod
    def entry_zip(cls, *args):
        return cls.entry_nary_op(lambda *x: tuple(x), *args)


class MetadataList(UserList):
    """
    class for working with a list of ScalarMetadata
    """

    # magic methods
    def __repr__(self):
        return f'{self.__class__.__name__}(len={len(self.data)}:d)'

    # general methods
    def append(self, item):
        if not isinstance(item, ScalarMetadata):
            raise TypeError(f'Only a {ScalarMetadata.__name__} object may be '
                            f'appended; got {type(item).__name__:s}.')
        super().append(item)

    def reduce_to_consistent(self, *, ignore_list=()) -> ScalarMetadata:
        consistent_metadata = ScalarMetadata()

        if not self:
            return consistent_metadata

        ignore_list = Metadata.key_list_to_tuple_key_list(ignore_list)

        first_metadata = self[0]
        for k in first_metadata.flat_key_set:
            if not (ignore_list and (k in ignore_list)):
                consistent_metadata[k] = first_metadata[k]

        for metadata in self[1:]:
            self.keep_consistent_metadata(consistent_metadata, metadata)

        return consistent_metadata

    # static methods
    @staticmethod
    def keep_consistent_metadata(metadata: ScalarMetadata,
                                 other: ScalarMetadata, /) -> None:
        """
        modifies medatata in-place
        """
        for k in metadata.different_keys(other):
            del metadata[k]

        for k in metadata.shared_keys(other):
            if metadata[k] != other[k]:
                del metadata[k]


def _key_to_description_key(k):
    """
    appends the description str to a key
    """
    return str(k) + DESCRIPTION_KEY_ADDON


def _make_time_str(d: dt.datetime):
    """
    creates a standardized time string from a datetime object
    """
    return d.strftime(DATETIME_FMT)


def _parse_time_str(s: str):
    """
    parses a standardized time string into a datetime object
    """
    return dt.datetime.strptime(s, DATETIME_FMT)


def _cast_xml_index(x: str | int) -> str:
    """
    converts a parsed xml value into an index string
    """
    x = _cast_xml_value(x)
    if isinstance(x, int):
        x = str(x)
    elif isinstance(x, str):
        pass
    else:
        raise TypeError(f'xml index must like a str or int; got '
                        f'{type(x).__name__}')
    return x


def _cast_xml_value(x: str) -> int | float | bool | str | None:
    """
    converts a parsed xml value into a python literal
    """
    assert isinstance(x, str), TypeError('Input must be a `str`.')

    try:
        return int(x)
    except ValueError:
        pass

    try:
        return float(x)
    except ValueError:
        pass

    x_lower = x.lower()
    if x_lower == str(True).lower():
        return True
    elif x_lower == str(False).lower():
        return False
    elif x_lower == str(None).lower():
        return None
    else:
        return x


def _pv_state_shard_to_dict(xml_element, as_entry=None):
    """
    takes a prairie view xml element and parses it into a dictionary
    """
    data_dict = dict()

    for child in xml_element:
        if child.tag == pvmk.PV_STATE_VALUE:
            key = child.attrib[pvmk.KEY]
            data_dict[key] = _parse_pv_state_value(child, as_entry=as_entry)
        else:
            raise ValueError(f'Unexpected tag ({child.tag}) encountered when '
                             f'parsing xml.')

    return data_dict


def _parse_pv_state_value(xml_element, as_entry=None):
    """
    parses an paririe view xml metadata element
    """
    if len(xml_element) == 0:
        v = _cast_xml_value(xml_element.attrib[pvmk.VALUE])
        return Entry(v) if as_entry else v
    else:
        indexed_vals = xml_element.findall(pvmk.INDEXED_VALUE)
        if indexed_vals:
            assert len(indexed_vals) == len(xml_element), \
                ValueError(f'`xml_element` had multiple types of child '
                           f'tags; expected only "{pvmk.INDEXED_VALUE}".')
            return _parse_indexed_values(indexed_vals, as_entry=as_entry)

        subindexed_vals = xml_element.findall(pvmk.SUBINDEXED_VALUES)
        if subindexed_vals:
            assert len(subindexed_vals) == len(xml_element), \
                ValueError(f'`xml_element` had multiple types of child '
                           f'tags; expected only "{pvmk.SUBINDEXED_VALUES}".')
            return _parse_subindexed_values(subindexed_vals, as_entry=as_entry)

        raise ValueError(f'Unexpected children of {pvmk.PV_STATE_VALUE} xml '
                         f'element "{xml_element.tag}" found.')


def _parse_subindexed_values(xml_elements, as_entry=None):
    """
    parses a prairie view xml metadata element with neseted indices
    """
    indexes = []
    values = []

    for e in xml_elements:
        indexes.append(_cast_xml_index(e.attrib[pvmk.INDEX]))

        assert len(e) > 0, ValueError(f'Expected {pvmk.SUBINDEXED_VALUES} '
                                      f'xml element to contain children.')

        v = _parse_indexed_values(e, index_key=pvmk.SUBINDEX, as_entry=as_entry)
        values.append(v)

    return dict(zip(indexes, values))


def _parse_indexed_values(xml_elements, index_key=pvmk.INDEX, as_entry=None):
    """
    parses a prairie view xml element with indexed values
    """
    indexes = []
    values = []
    descriptions = []
    has_descriptions = False

    for e in xml_elements:
        indexes.append(_cast_xml_index(e.attrib[index_key]))
        v = _cast_xml_value(e.attrib[pvmk.VALUE])
        values.append(Entry(v) if as_entry else v)

        if pvmk.DESCRIPTION in e.attrib:
            if not has_descriptions:
                has_descriptions = True
            descriptions.append(_cast_xml_value(e.attrib[pvmk.DESCRIPTION]))
        elif has_descriptions:
            raise ValueError('Expected xml element to have "description", but '
                             'none was found.')

    if has_descriptions:
        assert len(descriptions) == len(values), (
            'Expected descriptions and values lists to have the same number '
            'of elements.')

        # default to return a descriptive entry if as_entry is none
        if as_entry or (as_entry is None):
            for i, (description, entry) in enumerate(zip(descriptions, values)):
                if isinstance(entry, Entry):
                    value = entry.value
                else:
                    value = entry
                values[i] = DescriptiveEntry(value, description)
        else:
            indexes.extend([_key_to_description_key(idx) for idx in indexes])
            values.extend(descriptions)

    return dict(zip(indexes, values))


class _PVData(metaclass=abc.ABCMeta):
    """
    base class for handling any data from prairie view
    """

    __required_class_attribs = ['_indexing_metadata_properties',
                                '_child_class']
    _sort_metadata_for_debug = False

    # magic methods
    def __new__(cls, *args, **kwargs):
        if cls is _PVData:
            raise TypeError(f'{cls.__name__} class cannot be instantiated '
                            f'directly, it can only be used as a base class.')
        elif not issubclass(cls, _PVData):
            # checks if cls implements the necessary attributes
            # to determine why cls is not a subclass
            for a in _PVData.__required_class_attribs:
                if not hasattr(cls, a):
                    raise TypeError(f'Cannot instantiate `{cls.__name__}` '
                                    f'without class attribute `{a}` defined.')

            raise RuntimeError(f'{cls.__name__} is not a valid subclass of '
                               f'{_PVData.__name__}, check the subclass '
                               f'definition.')

        return super().__new__(cls)

    def __init__(self):
        super().__init__()

        self.raw_metadata: ScalarMetadata = ScalarMetadata()
        self.unique_metadata: ScalarMetadata = ScalarMetadata()
        self._child_list: PVDataList = PVDataList()

    def __repr__(self):
        if self._is_metadata_initialized:
            return f'{self.__class__.__name__}()'
        else:
            return f'uninitialized {self.__class__.__name__}'

    # interface definition
    @classmethod
    def __subclasshook__(cls, subclass):
        return (all([hasattr(subclass, a)
                     for a in _PVData.__required_class_attribs])
                and hasattr(subclass, '_is_metadata_initialized'))

    @property
    @abc.abstractmethod
    def _is_metadata_initialized(self) -> bool:
        return NotImplemented

    # general methods
    def get_metadata_prop(self, prop_name):
        return self.metadata[prop_name]

    def get_metadata_prop_value(self, prop_name):
        return self.metadata.get_value(prop_name)

    # protected methods
    def _update_child_list_from_xml(self, xml_elements):
        for e in xml_elements:
            new_child = self._child_class(xml_element=e, parent=self)
            self._child_list.append(new_child)

    def _initialize_unique_metadata(self,
                                    template_metadata: ScalarMetadata = None):
        if template_metadata is None:
            copied_template = ScalarMetadata.from_metadata(self.raw_metadata)
        else:
            copied_template = ScalarMetadata.from_metadata(template_metadata)
            copied_template.update_with_metadata(self.raw_metadata)

        if not self._child_list:
            # if there are no children of to this instance (i.e. at a
            # PV2DImage)
            self.unique_metadata = copied_template
            # self.unique_metadata = ScalarMetadata.from_dict(self.raw_metadata)
        else:
            # be sure the metadata of all children is sifted first ... this
            # ensures identification of unique metadata occurs from the lowest
            # level to the top level grabbing consistent metadata and bringing
            # it up along the way
            metadata_superlist = MetadataList()
            for child in self._child_list:
                # copying the metadata from high level to low level
                child._initialize_unique_metadata(copied_template)

                metadata_superlist.append(child.unique_metadata)

            self.unique_metadata = metadata_superlist.reduce_to_consistent(
                ignore_list=self._child_class._indexing_metadata_properties)

            if self._sort_metadata_for_debug:
                self.unique_metadata = self.unique_metadata.sorted()

            # only the consistent values will "rise back up" to become
            # objects unique metadata
            for child in self._child_list:
                child.unique_metadata.remove_templated_entries(
                    self.unique_metadata)
                child.unique_metadata.lock = True

    def _init_from_cache_helper(self, arrayed_dict):
        unique_dict = dict()
        child_dicts = []
        for k, v in arrayed_dict.items():
            if isinstance(v, list):
                if child_dicts:
                    for i, cd in enumerate(child_dicts):
                        cd[k] = v[i]
                else:
                    for entry in v:
                        child_dicts.append({k: entry})
            else:
                unique_dict[k] = v

        self.unique_metadata = ScalarMetadata.from_dict(
            unique_dict, check_for_tup_keys=True)
        self.unique_metadata.lock = True

        if self._child_class is None:
            assert not child_dicts, 'Expected `child_dicts` to be empty.'
            return

        for cd in child_dicts:
            new_child = self._child_class(parent=self)
            new_child._init_from_cache_helper(cd)
            self._child_list.append(new_child)

    # getters and setters
    @csutils.cached_property
    def metadata(self):
        if not self._is_metadata_initialized:
            log.warning('Warning: `metadata` property cannot be accessed prior '
                        'to metadata initialization. Returning `None`.')
            return None

        metadata = ArrayedMetadata.from_metadata(self.unique_metadata)
        for child in self._child_list:
            metadata.value_append_if_missing(child.metadata)

        if self._sort_metadata_for_debug:
            metadata = metadata.sorted()

        metadata.lock = True

        return metadata

    # class methods

    @classmethod
    def _child_class_tree(cls):
        if cls._child_class is None:
            return []
        else:
            return [cls._child_class, ] + cls._child_class._child_class_tree()


class PVDataList(UserList):
    """
    class for working with a list of _PVData objects
    """

    def filter_with_metadata_value(self, key, value):
        out_list = PVDataList()
        for pvd in self:
            if pvd.get_metadata_prop_value(key) == value:
                out_list.append(pvd)
        return out_list

    def filter_within_metadata_value_list(self, key, value_list):
        out_list = PVDataList()
        for pvd in self:
            if pvd.get_metadata_prop_value(key) in value_list:
                out_list.append(pvd)
        return out_list

    # magic methods
    def __repr__(self):
        return f'{self.__class__.__name__}(len={len(self.data)})'


class _PVDatasetChild(_PVData):
    """
    class for handing non-PVDataset (i.e. non-top-level) data from prairie view
    """

    # magic methods
    def __new__(cls, *args, **kwargs):
        if cls is _PVDatasetChild:
            raise TypeError(f'{cls.__name__} class cannot be instantiated '
                            f'directly, it can only be used as a base class. ')
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *, parent, xml_element=None):
        super().__init__()
        self._parent = parent
        if xml_element is not None:
            self._init_from_xml(xml_element)

    # protected methods
    def _init_from_xml(self, xml_element):
        for k, v in xml_element.attrib.items():
            self.raw_metadata[k] = _cast_xml_value(v)

    # getters and setters
    @property
    def _parent_tree(self):
        if isinstance(self._parent, _PVDatasetChild):
            return self._parent._parent_tree + [self._parent, ]
        else:
            return [self._parent, ]

    @property
    def _top_level_dataset(self) -> PVDataset:
        """this property represents the containing PVDataset"""
        return self._parent_tree[0]

    @csutils.cached_property
    def metadata(self):
        metadata = super().metadata

        if metadata is None:
            return None

        metadata.lock = False

        for p in self._parent_tree:
            metadata.update_with_metadata(p.unique_metadata)

        if self._sort_metadata_for_debug:
            metadata = metadata.sorted()

        metadata.lock = True

        return metadata

    @property
    def _is_metadata_initialized(self) -> bool:
        return self._top_level_dataset._is_metadata_initialized

    # class methods
    @classmethod
    def __subclasshook__(cls, subclass):
        return NotImplemented


class PV2DImage(_PVDatasetChild):
    """
    this class defines a single 2D image captured from the prairie view
    software with associated metadata

    in newer versions of PrairieView the image file path may point to a
    multi-page tiff (3D image) meaning that many frames can reference a
    single image file path
    """

    # class attributes
    _indexing_metadata_properties = [pvmk.CHANNEL_INDEX,
                                     pvmk.FILENAME,
                                     pvmk.FILE_PAGE,
                                     pvmk.TWO_D_IMAGE_ID]
    _child_class = None

    # magic methods
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_ndarray = None

    def __repr__(self):
        if self._is_metadata_initialized:
            return f'{self.__class__.__name__}(' \
                   f'two_d_image_id={self.two_d_image_id}' \
                   f'filename={self.filename}, ' \
                   f'page={self.file_page_index})'
        else:
            return super().__repr__()

    # protected methods
    def _init_from_xml(self, xml_element):
        super()._init_from_xml(xml_element)

        self.raw_metadata[pvmk.CHANNEL_INDEX] = (
                self.raw_metadata[pvmk.CHANNEL_NUM]
                - PV_CHANNEL_INDEX_CORRECTION)
        del self.raw_metadata[pvmk.CHANNEL_NUM]

        # generating the 2d image id
        two_d_image_id = self._top_level_dataset.last_2d_image_id
        if two_d_image_id is None:
            two_d_image_id = self._top_level_dataset.last_2d_image_id = 0
        else:
            two_d_image_id = self._top_level_dataset.last_2d_image_id = \
                two_d_image_id + 1
        self.raw_metadata[pvmk.TWO_D_IMAGE_ID] = two_d_image_id

        if self._sort_metadata_for_debug:
            self.raw_metadata = self.raw_metadata.sorted()

        self._top_level_dataset._all_key_set |= self.raw_metadata.flat_key_set
        self.raw_metadata.lock = True

    # general methods
    def load_image(self, *, force=False, with_memory_cache=False):
        if (not self.is_image_data_loaded) or force:
            file_image_data: np.ndarray
            if with_memory_cache:
                file_image_data = self._load_image_mem_cache(self.path)
                assert file_image_data is not None
            else:
                file_image_data = imread(self.path)
            if self.is_multipage:
                assert file_image_data.ndim == 3
                self.image_ndarray = file_image_data[self.file_page_index]
            else:
                assert file_image_data.ndim == 2
                self.image_ndarray = file_image_data
        else:
            log.debug('Image already loaded.')

    # Getters and Setters
    @property
    def two_d_image_id(self):
        return self.get_metadata_prop_value(pvmk.TWO_D_IMAGE_ID)

    @property
    def channel_index(self):
        return self.get_metadata_prop_value(pvmk.CHANNEL_INDEX)

    @property
    def path(self):
        return os.path.join(self._top_level_dataset.directory, self.filename)

    @property
    def filename(self):
        return self.get_metadata_prop_value(pvmk.FILENAME)

    @property
    def _raw_file_page_index(self) -> Optional[int]:
        try:
            return self.get_metadata_prop_value(pvmk.FILE_PAGE)
        except KeyError:
            # older PV versions will not have the "page" property
            return None

    @property
    def file_page_index(self) -> Optional[int]:
        raw_page_index = self._raw_file_page_index

        if raw_page_index is None:
            return raw_page_index

        return raw_page_index - PV_PAGE_INDEX_CORRECTION

    @property
    def is_multipage(self) -> bool:
        return self.file_page_index is not None

    @property
    def is_image_data_loaded(self) -> bool:
        return self.image_ndarray is not None

    # class methods
    _image_mem_cache: Dict[str, np.ndarray] = dict()

    @classmethod
    def _load_image_mem_cache(cls,
                              /,
                              path: Optional[str] = None,
                              *,
                              clear_cache: bool = False
                              ) -> Optional[np.ndarray]:
        if clear_cache:
            log.debug('Clearing {:s} load image cache.', cls.__name__)
            cls._image_mem_cache = dict()

        if path is None:
            return

        try:
            im_data = cls._image_mem_cache[path]
        except KeyError:
            log.debug('Did not find image file path in {:s} load image '
                      'cache ({:s}).',
                      cls.__name__, path)
            im_data = imread(path)
            cls._image_mem_cache[path] = im_data
        else:
            log.debug('Found image file path in {:s} load image cache ({:s}).',
                      cls.__name__, path)

        return im_data

    @classmethod
    def clear_load_image_cache(cls):
        cls._load_image_mem_cache(clear_cache=True)


class PVFrame(_PVDatasetChild):
    """
    this class defines a data structure for capturing a single scan from
    the prairie view software, essentially a multi-channel image (i.e.
    3D numeric array). Contains a list of `PV2DImage`s
    """

    # class attributes
    TRACK_NAME_PREFIX = 'Track'
    TRACK_ID_JOIN_STR = ', '

    _indexing_metadata_properties = [
        pvmk.FRAME_ID, pvmk.FRAME_INDEX, pvmk.TRACK_INDEX]
    _child_class = PV2DImage

    # magic methods

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        if self._is_metadata_initialized:
            return f'{self.__class__.__name__}(' \
                   f'frame_id={self.frame_id}, ' \
                   f'num_image_files={len(self.image_files)})'
        else:
            return super().__repr__()

    # protected methods

    def _init_from_xml(self, xml_element):
        super()._init_from_xml(xml_element)

        # handling the track index
        try:
            track_index = (self.raw_metadata[pvmk.TRACK]
                           - PV_TRACK_INDEX_CORRECTION)
        except KeyError:
            pass
        else:
            self.raw_metadata[pvmk.TRACK_INDEX] = track_index
            del self.raw_metadata[pvmk.TRACK]

            track_id_list = []
            try:
                parameter_set_name = self.raw_metadata[pvmk.PARAMETER_SET]
            except KeyError:
                pass
            else:
                track_id_list.append(parameter_set_name)

            track_id_list.append(
                f'{self.TRACK_NAME_PREFIX}{track_index}')

            # formats 'trackId' entry like:
            # '{parameter set name}, Track{track index}' for uniquely
            # identifying tracks belonging to different parameter sets
            track_id = self.TRACK_ID_JOIN_STR.join(track_id_list)
            self.raw_metadata[pvmk.TRACK_ID] = track_id

        # handling the frame index
        frame_index = (self.raw_metadata[pvmk.INDEX]
                       - PV_FRAME_INDEX_CORRECTION)
        self.raw_metadata[pvmk.FRAME_INDEX] = frame_index
        del self.raw_metadata[pvmk.INDEX]

        # generating the frame id
        frame_id = self._top_level_dataset.last_frame_id
        if frame_id is None:
            frame_id = self._top_level_dataset.last_frame_id = 0
        else:
            frame_id = self._top_level_dataset.last_frame_id = frame_id + 1
        self.raw_metadata[pvmk.FRAME_ID] = frame_id

        extra_params_element = xml_element.find(pvmk.EXTRA_PARAMETERS)
        if extra_params_element is not None:
            for k, v in extra_params_element.attrib.items():
                k = _cast_xml_index(k)
                if k in self.raw_metadata:
                    raise ValueError('A value in the "Frame" xml\'s "%s" '
                                     'element is already found in the '
                                     'frame\'s xml.', pvmk.EXTRA_PARAMETERS)
                new_entry = _cast_xml_value(v)
                self.raw_metadata[pvmk.CUSTOM_EXTRA_PARAMETERS, k] = new_entry

        pv_state_shard_element = xml_element.find(pvmk.PV_STATE_SHARD)
        if pv_state_shard_element is None:
            raise ValueError('Expected "Frame" xml element to have a '
                             '"PVStateShard" element as a child.')
        pv_state_shard_dict = _pv_state_shard_to_dict(pv_state_shard_element)
        pv_state_shard_metadata = ScalarMetadata.from_dict(pv_state_shard_dict)
        self.raw_metadata.update_with_metadata(pv_state_shard_metadata)

        if self._sort_metadata_for_debug:
            self.raw_metadata = self.raw_metadata.sorted()

        self._top_level_dataset._all_key_set |= self.raw_metadata.flat_key_set
        self.raw_metadata.lock = True

        # build the image file data list for this frame
        file_xml_elements = xml_element.findall(pvmk.FILE)
        self._update_child_list_from_xml(file_xml_elements)

    # getters and setters
    @property
    def frame_id(self):
        return self.get_metadata_prop_value(pvmk.FRAME_ID)

    @property
    def frame_index(self):
        return self.get_metadata_prop_value(pvmk.FRAME_INDEX)

    @property
    def track_index(self):
        return self.get_metadata_prop_value(pvmk.TRACK_INDEX)

    @property
    def track_id(self):
        return self.get_metadata_prop_value(pvmk.TRACK_ID)

    @property
    def image_files(self):
        return self._child_list

    @image_files.setter
    def image_files(self, x):
        self._child_list = x


class PVSequence(_PVDatasetChild):
    """
    this class defines a data structure for handling a list of
    `PVFrame`s. Part of the stack for importing images from prairie view
    """

    # class attributes
    _indexing_metadata_properties = [pvmk.SEQUENCE_INDEX]
    _child_class = PVFrame

    # magic methods
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        if self._is_metadata_initialized:
            return f'{self.__class__.__name__}(' \
                   f'sequence_index={self.sequence_index:d}, ' \
                   f'num_frames={len(self.frames):d})'
        else:
            return super().__repr__()

    # protected methods
    def _init_from_xml(self, xml_element):
        super()._init_from_xml(xml_element)

        raw_time_str = self.raw_metadata[pvmk.TIME]
        seq_level_time = dt.datetime.strptime(raw_time_str[:15], PV_TIME_FMT)
        # FIXME - this approach of using the min time could cause an error in
        #   in some edge cases. Might need to consider a different scheme.
        min_datetime = dt.datetime.min
        seq_level_time = seq_level_time.replace(
            year=min_datetime.year,
            month=min_datetime.month,
            day=min_datetime.day)
        self.raw_metadata[pvmk.TIME] = _make_time_str(seq_level_time)

        raw_cycle_index = self.raw_metadata[pvmk.CYCLE]
        cycle_index = raw_cycle_index - PV_CYCLE_INDEX_CORRECTION
        self.raw_metadata[pvmk.SEQUENCE_INDEX] = cycle_index
        del self.raw_metadata[pvmk.CYCLE]

        if self._sort_metadata_for_debug:
            self.raw_metadata = self.raw_metadata.sorted()

        self._top_level_dataset._all_key_set |= self.raw_metadata.flat_key_set
        self.raw_metadata.lock = True

        # build the frame data list for this sequence
        frame_xml_elements = xml_element.findall(pvmk.FRAME)
        self._update_child_list_from_xml(frame_xml_elements)

    # Getters and Setters

    @property
    def sequence_index(self):
        return self.get_metadata_prop_value(pvmk.SEQUENCE_INDEX)

    @property
    def frames(self):
        return self._child_list

    @frames.setter
    def frames(self, x):
        self._child_list = x


class PVDataset(_PVData):
    """
    This class defines the highest level of data from the prairie view software.
    This is the primary API for interacting and importing a PrairieView
    data directory. Contains a list of `PVSequence`s.
    """

    # class attributes
    _indexing_metadata_properties = []
    _child_class = PVSequence

    last_frame_id = None
    last_2d_image_id = None
    verbose = False

    # magic methods
    def __init__(self,
                 path: str,
                 /,
                 *,
                 use_cache: bool = True,
                 name: str = None,
                 lazy: bool = False,
                 clear_cache: bool = False,
                 clear_cache_and_exit: bool = False,
                 parallel_load: bool = True):
        super().__init__()

        # primary to directory for the dataset
        self.directory: str = None

        # pretty name for the dataset
        self.name: str = name

        # setup hidden properties
        self._category = None
        self._xml_filename: str = None
        self._all_key_set = set()
        # prevents accessing metadata before loading from xml, sifting and
        # initialization
        self.__did_initialize_metadata = False

        # determine the initialization to use based on what's found at the
        # `path` argument to initialize the dataset
        self._init_from_path(path,
                             use_cache,
                             lazy,
                             clear_cache,
                             clear_cache_and_exit,
                             parallel_load)

    # this method needs to be declared since _PVData is an abstract base
    # metaclass if we want `isinstance()` to behave properly
    @classmethod
    def __subclasshook__(cls, subclass):
        return NotImplemented

    def __repr__(self):
        if self._is_metadata_initialized:
            name_str = f'\'{self.name}\'' if self.name is not None else 'None'
            return f'{self.__class__.__name__}(' \
                   f'name={name_str}, ' \
                   f'num_sequences={len(self.sequences):d})'
        else:
            return super().__repr__()

    def __bool__(self):
        return self.num_2d_images > 0

    # protected methods
    def _init_from_path(self,
                        path,
                        use_cache,
                        lazy,
                        clear_cache,
                        clear_cache_and_exit,
                        parallel_load: bool):
        init_function = None
        init_func_kwargs = dict()
        init_func_kwargs['lazy'] = lazy
        init_func_kwargs['parallel_load'] = parallel_load

        clear_cache = clear_cache or clear_cache_and_exit

        if type(path) is str:
            if os.path.isdir(path):
                self.directory = path

                csutils.touchdir(self._cache_dir)

                if os.path.isfile(self._cache_path) and use_cache \
                        and (not clear_cache):
                    init_function = self._init_from_cache

                glob_list_xml = glob.glob(os.path.join(path,
                                                       '*' + XML_EXTENSION))
                if len(glob_list_xml) == 1:
                    _, self._xml_filename = os.path.split(glob_list_xml[0])
                    if init_function is None:
                        init_function = self._init_from_xml
                        init_func_kwargs['do_cache'] = use_cache
                    else:
                        init_func_kwargs['with_xml_fallback'] = True

                if init_function is None:
                    if len(glob_list_xml) == 0:
                        msg = (f'Passed directory does not contain an '
                               f'{XML_EXTENSION} file. Looking in "{path}".')
                        raise ValueError(msg)
                    else:
                        msg = (f'Passed directory contains multiple '
                               f'{XML_EXTENSION} files. Looking in "{path}".')
                        raise ValueError(msg)

            elif os.path.isfile(path):
                if path.lower().endswith(PICKLE_EXTENSION):
                    init_function = self._init_from_cache
                    init_func_kwargs['cache_path'] = path
                elif path.lower().endswith(XML_EXTENSION):
                    init_function = self._init_from_xml
                    init_func_kwargs['do_cache'] = use_cache
                    self.directory, self._xml_filename = os.path.split(path)
                else:
                    msg = (f'Expected "{XML_EXTENSION}" or "{PICKLE_EXTENSION}"'
                           f'-pvdataset file path to be passed; none found.')
                    raise FileNotFoundError(msg)
            else:
                msg = (f'Could not build dataset from passed string; must be a '
                       f'PrairieView dataset directory or path to a '
                       f'{XML_EXTENSION} or {PICKLE_EXTENSION}-dataset file.')
                raise ValueError(msg)
        else:
            msg = (f'str of an XML path, H5 dataset path, or a PrairieView '
                   f'image directory path must be passed; got {repr(path)}.')
            raise TypeError(msg)

        if clear_cache:
            self.clear_cache()
            if clear_cache_and_exit:
                log.info('Cache for dataset has been cleared. '
                         'Exiting initialization.')
                return

        init_function(**init_func_kwargs)

    def _init_from_xml(self,
                       *,
                       lazy,
                       parallel_load: bool,
                       do_cache=True):
        log.info('Importing dataset metadata from xml.')

        log.debug('Loading the top-level dataset metadata.')
        # load the top-level dataset metadata
        top_level_metadata_xml = self._xml.find(pvmk.PV_STATE_SHARD)
        self.raw_metadata = self.raw_metadata.from_dict(
            _pv_state_shard_to_dict(top_level_metadata_xml.root))

        # convert timestamp to a more consistently formatted string
        top_level_timestamp = dt.datetime.strptime(
            self._xml.root.attrib[pvmk.DATE], PV_DATETIME_FMT)
        self.raw_metadata[pvmk.TIMESTAMP] = _make_time_str(top_level_timestamp)

        self.raw_metadata[pvmk.NOTES] = self._xml.root.attrib[pvmk.NOTES]

        if self._sort_metadata_for_debug:
            self.raw_metadata = self.raw_metadata.sorted()

        self._all_key_set |= self.raw_metadata.flat_key_set
        self.raw_metadata.lock = True

        # load the lower levels of metadata by building the sequence data list
        # for this dataset
        log.debug('Loading metadata for all sequences in the dataset.')
        sequence_xml_elements = self._xml.root.findall(pvmk.SEQUENCE)
        self._update_child_list_from_xml(sequence_xml_elements)

        # initialize the unique metadata for each pv_data object in the dataset
        # including self, running this method is required for getting
        # the computed attribute `metadata` to operate properly
        self._sift_metadata()

        if do_cache:
            self.cache_to_disk(lazy=lazy, parallel_load=parallel_load)

    def _init_from_cache(self,
                         *,
                         lazy: bool,
                         parallel_load: bool,
                         cache_path: Optional[str] = None,
                         with_xml_fallback: bool = False):
        try:
            log.info('Attempting to load dataset from cache.')
            if cache_path is None:
                cache_path = self._cache_path

            with open(cache_path, 'rb') as _file:
                cache_object: _PVDatasetCache = pickle.load(_file)

            log.debug('Cache-file loaded.')
            self.name = cache_object.name
            self._category = cache_object.category

            cache_metadata_dict = cache_object.metadata

            log.debug('Initializing metadata from cache-file.')
            self._init_from_cache_helper(cache_metadata_dict)
            self.__did_initialize_metadata = True

            log.info('Loading of dataset from cache was successful.')
        except Exception as e:
            log.error('Initialization of dataset from cache failed.',
                      exc_info=e)
            if with_xml_fallback:
                log.info('Falling back to initialize dataset from xml.')
                self._init_from_xml(lazy=lazy, parallel_load=parallel_load)
                return

            raise

        did_modify_im_data = False
        if cache_object.images is not None:
            try:
                log.info('Loading in cached images.')
                assert len(self.all_2d_images) == len(cache_object.images), (
                    'Unexpected length mismatch between image files and saved '
                    'images.')
                iterator = zip(self.all_2d_images, cache_object.images)
                did_modify_im_data = True
                for im_file, im in iterator:
                    im_file.image_ndarray = im
                log.info('Loading in of cached images is sucessful.')
            except Exception as e:
                log.warning('Could not read-in cached image data.',
                            exc_info=e)
                if did_modify_im_data:
                    log.debug('Clearing any potentially faulty cached image '
                              'data.')
                    for im_file in self.all_2d_images:
                        im_file.image_ndarray = None

    # @csutils.timed(logger=log)
    def _sift_metadata(self):
        """
        this function looks through the metadata dicts for all child objects
        and moves any consistent values up to this object's metadata
        """
        log.debug('Sifting the dataset\'s metadata; this may take some time.')
        # create a template metadata object that has the keys for every possible
        # entry. This way up and down the tree at any _PVData object within
        # the dataset the same set of keys is present.
        log.debug('Creating a template metadata set.')
        template_metadata = ScalarMetadata()
        for k in self._all_key_set:
            template_metadata[k] = EmptyEntry()

        log.debug('Initializing unique metadata for all _PVData objects.')
        self._initialize_unique_metadata(template_metadata)
        self.unique_metadata.lock = True
        log.debug('Unique metadata initialization complete; all dataset '
                  'metadata is successfully sifted.')
        self.__did_initialize_metadata = True

    # general methods

    def clear_cache(self):
        if self.directory is None:
            log.warning('Cannot clear cache because a directory is not'
                        'associated with this dateset.')
        else:
            log.debug('Deleting cache file at: "{}"', self._cache_path)
            try:
                os.remove(self._cache_path)
            except FileNotFoundError as e:
                log.debug('Cache file not found.', exc_info=e)
                log.info('Cache file not found.')
            else:
                log.info('Cache file deleted.')

    def cache_to_disk(self, *, lazy=False, parallel_load=True):
        log.info('Caching dataset to disk.')
        cache_object = _PVDatasetCache(self,
                                       lazy=lazy,
                                       parallel_load=parallel_load)

        log.debug('Saving cache of dataset to disk at "{}".',
                  self._cache_path)
        if os.path.exists(self._cache_path):
            log.debug('Overwriting existing cache data.')
        with open(self._cache_path, 'wb') as _file:
            pickle.dump(cache_object, _file)

    # @csutils.timed(log)
    def load_all_image_data(self, *, do_parallel=True, force=False):
        """
        loads the image for every PV2DImage object of the dataset
        """
        all_2d_images = self.all_2d_images
        num_2d_images = len(all_2d_images)

        # we need to handle the case where multiple 2d-images are stored in a
        # single file, so we will create a mapping where each key is a file path
        # and it's values are a list of 2D image objects which reference that
        # file path
        path_to_2d_im_map: Dict[str, List[PV2DImage]] = dict()
        for pv_2d_im in all_2d_images:
            pv_2d_im: PV2DImage
            im_path = pv_2d_im.path
            try:
                path_to_2d_im_map[im_path].append(pv_2d_im)
            except KeyError:
                path_to_2d_im_map[im_path] = [pv_2d_im]
        num_unique_file_paths = len(path_to_2d_im_map)

        log.info('Loading up to {:d} 2D image(s) from {:d} image files on '
                 'disk; this may take some time.',
                 num_2d_images, num_unique_file_paths)

        if do_parallel:
            image_paths_to_load: Set[str] = set()
            pv_2d_images_to_load: List[PV2DImage] = []
            for pv_2d_im in all_2d_images:
                if (not pv_2d_im.is_image_data_loaded) or force:
                    image_paths_to_load.add(pv_2d_im.path)
                    pv_2d_images_to_load.append(pv_2d_im)

            # read in all the images in parallel using the max number of threads
            if image_paths_to_load:
                num_threads = os.cpu_count()
                log.info('Loading images using {:d} parallel processes.',
                         num_threads)
                # FIXME - too much nesting
                with Pool(num_threads) as p:
                    images = p.imap(imread, image_paths_to_load)
                    for im_path, image_arr in zip(image_paths_to_load, images):
                        im_path: str
                        image_arr: np.ndarray

                        for pv_2d_im in path_to_2d_im_map[im_path]:
                            if pv_2d_im not in pv_2d_images_to_load:
                                continue

                            if pv_2d_im.is_multipage:
                                assert (image_arr.ndim == 3)
                                pv_2d_im.image_ndarray \
                                    = image_arr[pv_2d_im.file_page_index]
                            else:
                                assert (image_arr.ndim == 2)
                                pv_2d_im.image_ndarray = image_arr
            else:
                log.info('All image files already loaded.')
                return
        else:
            for pv_2d_im in all_2d_images:
                pv_2d_im.load_image(force=force, with_memory_cache=True)
            PV2DImage.clear_load_image_cache()

        log.info('All images loaded.')

    def get_all_image_data(self, *, lazy=False, **kwargs):
        if not lazy:
            self.load_all_image_data(**kwargs)
        return [im_file.image_ndarray for im_file in self.all_2d_images]

    @property
    def category(self):
        """
        determine the PVDatasetCategory of the dataset, pre-flagging step
        before conversion of the dataset into an ImagingDataset
        """
        if self._category is not None:
            return self._category

        if self.verbose:
            log.info('Attempting to determine the PVDataset parsing schema.')

        if self.num_2d_images == 1:
            log.info('Dataset can be categorized as a {}',
                     PVDatasetCategory.SINGLE_IMAGE)
            self._category = PVDatasetCategory.SINGLE_IMAGE
            return self._category

        if len(self.all_frames) == 1:
            log.info('Dataset can be categorized as a {}',
                     PVDatasetCategory.MULTI_CHANNEL_SINGLE_IMAGE)
            self._category = PVDatasetCategory.MULTI_CHANNEL_SINGLE_IMAGE
            return self._category

        md_reader = PVMetadataInterpreter(self.metadata)
        full_category = PVDatasetCategory.UNKNOWN

        if max(list(md_reader.get_num_channels_by_frame().values())) > 1:
            # FIXME - if all frames have the channel, but the channel
            #  index is not consistent the category of such a dataset will not
            #  be classified as multi-channel by this criterion.
            full_category |= PVDatasetCategory.MULTI_CHANNEL

        if not md_reader.is_channel_index_by_frame_consistent():
            log.warning('The number of channels is not consistent across all '
                        'frames; likely cannot categorize this PVDataset.')

        if md_reader.has_tracks:
            full_category |= PVDatasetCategory.MULTI_TRACK

        if len(self.sequences) > 1:
            full_category |= PVDatasetCategory.MULTI_SEQUENCE

        # criteria used for selecting dataset types
        xy_size_crit = Criterion(
            md_reader.is_xy_size_consistent(),
            'was not acquired with a consistent x-y scan size/resolution.')
        xy_pitch_crit = Criterion(
            md_reader.is_xy_pitch_consistent(),
            'was not acquired with a consistent x-y scan pitch/resolution.')
        xy_pos_crit = Criterion(
            md_reader.is_xy_position_consistent(),
            'was not acquired with a consistent x-y position.')
        im_ids_by_seq_crit = Criterion(
            md_reader.is_num_2d_images_by_sequence_consistent(),
            'does not have consistent number of image 2d images in each '
            'sequence.')
        im_ids_by_z_crit = Criterion(
            md_reader.is_num_2d_images_by_net_z_location_consistent(),
            'does not have a consistent number of 2d images for each net z-'
            'location.')
        no_tracks_crit = Criterion(
            not md_reader.has_tracks, 'has parameter tracks.')
        tracks_by_z_crit = Criterion(
            (not md_reader.has_tracks)
            or md_reader.is_track_index_by_net_z_location_consistent(),
            'does not have consistent track indexes across net z-locations.')
        z_spacing_crit = Criterion(
            (md_reader.has_multiple_net_z_locations and
             md_reader.is_net_z_location_spacing_consistent(
                 sorting=MetadataSorting.ACQUISITION)),
            'does not have consistent z-spacing (by acquisition).')
        channel_index_crit = Criterion(
            md_reader.is_channel_index_by_frame_consistent(),
            'does not have consistent channel index(es) across frames.')
        forced_z_spacing_crit = Criterion(
            (md_reader.has_multiple_net_z_locations and
             md_reader.is_net_z_location_spacing_consistent(
                 sorting=MetadataSorting.VALUE)),
            'does not have consistent z-spacing across z-values.')
        track_id_crit = Criterion(
            (not md_reader.has_tracks)
            or md_reader.is_track_id_by_net_z_location_consistent(),
            'does not have consistent track ids at each net z-location.')
        single_im_id_per_seq_crit = Criterion(
            md_reader.is_num_2d_images_by_sequence_unity(),
            'does not have a one and only one 2d image for each sequence.')
        single_im_id_per_z_crit = Criterion(
            md_reader.is_num_2d_images_by_net_z_location_unity(),
            'does not have one and only one 2d image per net z-location.')
        single_sequence_crit = Criterion(
            md_reader.num_sequences == 1,
            'does not have one and only one sequence.')
        multi_sequence_crit = Criterion(
            md_reader.num_sequences > 1,
            'does not have more than one sequence.')
        im_ids_per_frame_crit = Criterion(
            md_reader.is_num_2d_images_by_frame_consistent(),
            'does not have consistent number of 2d images at each frame.')
        multi_z_crit = Criterion(
            md_reader.has_multiple_net_z_locations,
            'does not have multiple net-z locations.')

        # criterion for a t-series
        t_series_crit = CriteriaList(
            [single_sequence_crit,
             im_ids_per_frame_crit, no_tracks_crit, channel_index_crit,
             xy_pos_crit, xy_pitch_crit, xy_size_crit])

        if t_series_crit:
            if self.verbose:
                log.info('Dataset can be categorized as {}',
                         PVDatasetCategory.T_SERIES)

            warn_crit = CriteriaList([])
            warn_crit.check_crits_and_log(
                message_prefix=f'Warning: PVDataset can be categorized as '
                               f'{PVDatasetCategory.T_SERIES}, but PVDataset',
                log_fcn=log.warning if self.verbose else log.debug)

            full_category |= PVDatasetCategory.T_SERIES
        else:
            t_series_crit.check_crits_and_log(
                message_prefix='PVDataset cannot be categorized as a T-series '
                               'because PVDataset ',
                log_fcn=log.info if self.verbose else log.debug)

        # criterion for a t-z-series
        tz_series_crit = CriteriaList(
            [im_ids_by_seq_crit, im_ids_by_z_crit, no_tracks_crit,
             channel_index_crit, xy_pos_crit, xy_pitch_crit, xy_size_crit,
             multi_sequence_crit, multi_z_crit])

        if tz_series_crit:
            if self.verbose:
                log.info('Dataset can be categorized as a {}',
                         PVDatasetCategory.TZ_SERIES_IMAGE)

            warn_crit = CriteriaList(
                [z_spacing_crit, forced_z_spacing_crit, track_id_crit])
            warn_crit.check_crits_and_log(
                message_prefix='Warning: PVDataset can be categorized as a TZ '
                               'Series, but PVDataset ',
                log_fcn=log.warning if self.verbose else log.debug)
            # FIXME - don't hard code category name into message prefix
            # FIXME - make this if else it's own statement variable

            full_category |= PVDatasetCategory.TZ_SERIES_IMAGE

        else:
            tz_series_crit.check_crits_and_log(
                message_prefix='PVDataset cannot be categorized as a TZ '
                               'Series because PVDataset ',
                log_fcn=log.info if self.verbose else log.debug)

        # criterion for a z-series
        z_series_crit = CriteriaList(
            [im_ids_by_z_crit, tracks_by_z_crit, channel_index_crit,
             xy_pos_crit, xy_pitch_crit, xy_size_crit, multi_z_crit])

        if z_series_crit:
            if self.verbose:
                log.info('Dataset can be categorized as a {}',
                         PVDatasetCategory.Z_STACK_IMAGE)

            warn_crit = CriteriaList(
                [z_spacing_crit, forced_z_spacing_crit, track_id_crit])
            warn_crit.check_crits_and_log(
                message_prefix='Warning: PVDataset can be categorized as a '
                               'Z stack dataset, but PVDataset ',
                log_fcn=log.warning if self.verbose else log.debug)

            full_category |= PVDatasetCategory.Z_STACK_IMAGE
        else:
            z_series_crit.check_crits_and_log(
                message_prefix='PVDataset cannot be categorized as a '
                               'Z-stack dataset because PVDataset ',
                log_fcn=log.info if self.verbose else log.debug)

        self._category = full_category
        return self._category

    # getters and setters
    @property
    def _is_metadata_initialized(self) -> bool:
        return self.__did_initialize_metadata

    @property
    def _xml(self):
        return csutils.XML(self._xml_path)

    @property
    def _xml_path(self):
        return os.path.join(self.directory, self._xml_filename)

    @property
    def _cache_dir(self):
        return os.path.join(self.directory, CACHE_DIR_NAME)

    @property
    def _cache_path(self):
        return os.path.join(self._cache_dir, _PVDatasetCache.CACHE_FILENAME)

    @property
    def sequences(self):
        return self._child_list

    @sequences.setter
    def sequences(self, x):
        self._child_list = x

    @property
    def all_2d_images(self) -> PVDataList[PV2DImage]:
        return PVDataList([im_file
                           for frame in self.all_frames
                           for im_file in frame.image_files])

    @property
    def num_2d_images(self):
        return len(self.all_2d_images)

    @property
    def all_frames(self):
        return PVDataList([frame
                           for seq in self.sequences for frame in seq.frames])

    @property
    def all_child_pvdata_objects(self):
        return self.sequences + self.all_frames + self.all_2d_images


def _set_all_nested_dict_values(nested_dict, val):
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            _set_all_nested_dict_values(v, val)
        else:
            nested_dict[k] = val


class PVDatasetList(PVDataList):
    """
    wrapper class for handling a list of PVDataset objects

    entries should either be of type PVDataset or NoneType
    """

    def get_by_name(self, dataset_name):
        for d in self.data:
            if d is None:
                continue

            if d.name == dataset_name:
                return d

        msg = (f'Dataset with name \'{dataset_name}\' not found in the dataset '
               f'list.')
        raise ValueError(msg)


class _PVDatasetCache(object):
    """
    helper class for dumping/restoring a PVDataset object to/from the disk
    """

    CACHE_FILENAME = '.py_pvdataset_cache' + PICKLE_EXTENSION

    def __init__(self, dataset: PVDataset, *, lazy, parallel_load):
        self.name = dataset.name
        self.metadata = dataset.metadata.data
        if lazy:
            self.category = dataset._category
        else:
            self.category = dataset.category
        self._images = dataset.get_all_image_data(
            lazy=lazy,
            do_parallel=parallel_load)
        self.process_image_data()

    def process_image_data(self):
        if not self._images:
            self._images = None
            return

        for im in self._images:
            if im is None:
                return

        ref_im = self._images[0]
        ref_shape = ref_im.shape

        for im in self._images:
            if not (im.shape == ref_shape):
                return

        stacked_array_shape = (len(self._images),) + ref_shape
        stack_array = np.zeros_like(ref_im, shape=stacked_array_shape)
        np.stack(self._images, out=stack_array)
        self._images = stack_array

    @property
    def images(self):
        return self._images


class MetadataSorting(enum.Enum):
    """
    enumeration class specifying how to sort metadata entries
    """

    ACQUISITION = enum.auto()
    VALUE = enum.auto()

    def __repr__(self):
        return f'{self.__class__.__name__}.{self.name}'


class PVDatasetCategory(enum.Flag):
    """
    flag class for specifying the category of a PVDataset
    """

    UNKNOWN = 0
    SINGLE_IMAGE = enum.auto()
    MULTI_CHANNEL = enum.auto()
    MULTI_CHANNEL_SINGLE_IMAGE = SINGLE_IMAGE | MULTI_CHANNEL
    MULTI_TRACK = enum.auto()
    MULTI_SEQUENCE = enum.auto()
    T_SERIES = enum.auto()
    TZ_SERIES_IMAGE = enum.auto()
    Z_STACK_IMAGE = enum.auto()


class PVMetadataInterpreter(object):
    """
    class for interpreting a Metadata object belonging to a PVDataset instance

    this class looks through Metadata to determine different properties of the
    datasets images and evaluate their consistency throughout the dataset
    """

    DIFF_THRESHOLD = 1e-6

    X_POSITION_KEY = (pvmk.POSITION_CURRENT, pvmk.X_AXIS, '0')
    Y_POSITION_KEY = (pvmk.POSITION_CURRENT, pvmk.Y_AXIS, '0')
    Z_POSITION_KEY = (pvmk.POSITION_CURRENT, pvmk.Z_AXIS)
    POCKELS_KEY = (pvmk.LASER_POWER, MAIN_LASER_POCKELS_INDEX)

    _no_net_z_msg = 'No net z locations found in metadata.'
    _no_seq_msg = 'No sequence indexes found in metadata.'
    _no_frame_id_msg = 'No frame ids found in metadata.'

    # magic methods
    def __init__(self, metadata: Metadata):
        if not isinstance(metadata, Metadata):
            raise TypeError(
                f'{self.__class__.__name__} object must be instantiated with '
                f'a {Metadata.__name__} object, got an object of type '
                f'{metadata.__class__.__name__} instead.')

        if not metadata.lock:
            msg = (f'Metadata object must be locked to create a '
                   f'{self.__class__.__name__} object.')
            raise ValueError(msg)

        super().__init__()
        self.metadata = metadata

    # general methods
    @csutils.cache
    def get_unique_net_z_locations(self,
                                   *,
                                   sorting=MetadataSorting.ACQUISITION):
        net_z_locs = ArrayedMetadata.linearize_entry(self.net_z_location)
        if sorting is MetadataSorting.ACQUISITION:
            # FIXME - this will not catch the case where a value appears
            #  twice outside of a monotonic ordering
            unique_locs = []
            for val in net_z_locs:
                if val not in unique_locs:
                    unique_locs.append(val)
        elif sorting is MetadataSorting.VALUE:
            unique_locs = sorted(list(set(net_z_locs)))
        else:
            raise RuntimeError(f'Sorting scheme "{sorting}" not recognized.')
        return unique_locs

    @csutils.cache
    def get_net_z_location_spacing(self,
                                   *,
                                   sorting=MetadataSorting.ACQUISITION):
        z_locs = np.array(self.get_unique_net_z_locations(sorting=sorting))
        z_loc_spacing = np.diff(z_locs)
        return z_loc_spacing.tolist()

    @csutils.cache
    def get_net_z_location_by_sequence(self):
        return self.get_entry_by_entry(self.sequence_index,
                                       self.net_z_location)

    @csutils.cache
    def get_metadata_value_by_net_z_location(self, metadata_key):
        return self.get_entry_by_entry(self.net_z_location,
                                       self.metadata[metadata_key])

    @csutils.cache
    def get_track_id_by_net_z_location(self):
        return self.get_metadata_value_by_net_z_location(pvmk.TRACK_ID)

    @csutils.cache
    def get_track_index_by_net_z_location(self):
        return self.get_metadata_value_by_net_z_location(pvmk.TRACK_INDEX)

    @csutils.cache
    def get_num_tracks_by_net_z_locations(self):
        track_index_dict = self.get_track_index_by_net_z_location()
        return self.list_dict_to_len_dict(track_index_dict)

    @csutils.cache
    def get_filenames_by_net_z_location(self):
        self.filename_ref_deprecation_warning()
        return self.get_metadata_value_by_net_z_location(pvmk.FILENAME)

    @csutils.cache
    def get_im_ids_by_net_z_location(self):
        return self.get_metadata_value_by_net_z_location(pvmk.TWO_D_IMAGE_ID)

    @csutils.cache
    def get_filenames_by_sequence(self):
        self.filename_ref_deprecation_warning()
        return self.get_entry_by_entry(self.sequence_index,
                                       self.filename)

    @csutils.cache
    def get_im_ids_by_sequence(self):
        return self.get_entry_by_entry(self.sequence_index,
                                       self.two_d_image_id)

    @csutils.cache
    def get_filenames_by_frame_id(self):
        self.filename_ref_deprecation_warning()
        return self.get_entry_by_entry(self.frame_id,
                                       self.filename)

    @csutils.cache
    def get_im_ids_by_frame_id(self):
        return self.get_entry_by_entry(self.frame_id,
                                       self.two_d_image_id)

    @csutils.cache
    def get_num_files_by_net_z_location(self):
        self.filename_ref_deprecation_warning()
        filename_dict = self.get_filenames_by_net_z_location()
        return self.list_dict_to_len_dict(filename_dict)

    @csutils.cache
    def get_num_2d_images_by_net_z_location(self):
        im_id_dict = self.get_im_ids_by_net_z_location()
        return self.list_dict_to_len_dict(im_id_dict)

    @csutils.cache
    def get_num_files_by_sequence(self):
        self.filename_ref_deprecation_warning()
        filename_dict = self.get_filenames_by_sequence()
        return self.list_dict_to_len_dict(filename_dict)

    @csutils.cache
    def get_num_2d_images_by_sequence(self):
        im_id_dict = self.get_im_ids_by_sequence()
        return self.list_dict_to_len_dict(im_id_dict)

    @csutils.cache
    def get_num_files_by_frame_id(self):
        self.filename_ref_deprecation_warning()
        filename_dict = self.get_filenames_by_frame_id()
        return self.list_dict_to_len_dict(filename_dict)

    @csutils.cache
    def get_num_2d_images_by_frame_id(self):
        im_id_dict = self.get_im_ids_by_frame_id()
        return self.list_dict_to_len_dict(im_id_dict)

    @csutils.cache
    def get_channel_index_by_frame(self):
        return self.get_entry_by_entry(self.frame_id,
                                       self.channel_index)

    @csutils.cache
    def get_num_channels_by_frame(self):
        channel_index_dict = self.get_channel_index_by_frame()
        return self.list_dict_to_len_dict(channel_index_dict)

    @csutils.cache
    def get_channel_name_by_channel_index(self):
        return self.get_entry_by_entry(self.channel_index,
                                       self.channel_name)

    @csutils.cache
    def get_relative_time_by_sequence_index(self):
        return self.get_entry_by_entry(by=self.sequence_index,
                                       get=self.relative_time)

    @csutils.cache
    def get_starting_relative_time_by_sequence_index(self):
        rel_time_by_seq = self.get_relative_time_by_sequence_index()
        out_dict = dict()
        for k, v in rel_time_by_seq.items():
            out_dict[k] = min(v)
        return out_dict

    # TODO - consider if the `is...` methods should always return true or false
    #  and catch / not raise any errors. Maybe we should use a wrapper function
    #  which can take a `do_fail` flag ...
    @csutils.cache
    def is_num_files_by_net_z_location_consistent(self):
        self.filename_ref_deprecation_warning()
        num_f_dict = self.get_num_files_by_net_z_location()
        num_f_list = list(num_f_dict.values())

        if not num_f_list:
            _msg = (self._no_net_z_msg +
                    ' Cannot compute if num files is consistent across net '
                    'z locations. `num_f_list` is empty.')
            raise ValueError(_msg)

        return self.is_value_consistent(num_f_list)

    @csutils.cache
    def is_num_2d_images_by_net_z_location_consistent(self):
        num_im_dict = self.get_num_2d_images_by_net_z_location()
        num_im_list = list(num_im_dict.values())

        if not num_im_list:
            _msg = (self._no_net_z_msg +
                    ' Cannot compute if num 2d images is consistent across net '
                    'z locations. `num_im_list` is empty.')
            raise ValueError(_msg)

        return self.is_value_consistent(num_im_list)

    @csutils.cache
    def is_num_files_by_sequence_consistent(self):
        self.filename_ref_deprecation_warning()
        num_f_dict = self.get_num_files_by_sequence()
        num_f_list = list(num_f_dict.values())

        if not num_f_list:
            _msg = (self._no_seq_msg +
                    ' Cannot compute if num files is consistent across '
                    ' sequences. `num_f_list` is empty.')
            raise ValueError(_msg)

        return self.is_value_consistent(num_f_list)

    @csutils.cache
    def is_num_2d_images_by_sequence_consistent(self):
        num_im_dict = self.get_num_2d_images_by_sequence()
        num_im_list = list(num_im_dict.values())

        if not num_im_list:
            _msg = (self._no_seq_msg +
                    ' Cannot compute if num 2d images is consistent across '
                    ' sequences. `num_im_list` is empty.')
            raise ValueError(_msg)

        return self.is_value_consistent(num_im_list)

    @csutils.cache
    def is_num_files_by_sequence_unity(self):
        self.filename_ref_deprecation_warning()
        if self.is_num_files_by_sequence_consistent():
            # FIXME - develop a method to read out an example value from an
            #  entry .... maybe need to have the `get_by...` function work as a
            #  generator ... yielding elements. ... but it's a dict ...
            num_f_list = list(self.get_num_files_by_sequence().values())
            return num_f_list[0] == 1
        else:
            return False

    @csutils.cache
    def is_num_2d_images_by_sequence_unity(self):
        if not self.is_num_2d_images_by_sequence_consistent():
            return False

        # FIXME - develop a method to read out an example value from an
        #  entry .... maybe need to have the `get_by...` function work as a
        #  generator ... yielding elements. ... but it's a dict ...
        num_im_list = list(self.get_num_2d_images_by_sequence().values())
        return num_im_list[0] == 1

    @csutils.cache
    def is_num_files_by_net_z_location_unity(self):
        self.filename_ref_deprecation_warning()
        if self.is_num_files_by_net_z_location_consistent():
            num_f_list = list(self.get_num_files_by_sequence().values())
            return num_f_list[0] == 1
        else:
            return False

    @csutils.cache
    def is_num_2d_images_by_net_z_location_unity(self):
        if not self.is_num_2d_images_by_net_z_location_consistent():
            return False

        num_im_list = list(self.get_num_2d_images_by_sequence().values())
        return num_im_list[0] == 1

    def is_num_files_by_frame_consistent(self):
        self.filename_ref_deprecation_warning()
        num_f_dict = self.get_num_files_by_frame_id()
        num_f_list = list(num_f_dict.values())

        if not num_f_list:
            msg = (self._no_seq_msg +
                   ' Cannot compute if num files is consistent across '
                   'sequences. `num_f_list` is empty.')
            raise ValueError(msg)

        return self.is_value_consistent(num_f_list)

    def is_num_2d_images_by_frame_consistent(self):
        num_im_dict = self.get_num_2d_images_by_frame_id()
        num_im_list = list(num_im_dict.values())

        if not num_im_list:
            msg = (self._no_seq_msg +
                   ' Cannot compute if num 2d images is consistent across '
                   'sequences. `num_im_list` is empty.')
            raise ValueError(msg)

        return self.is_value_consistent(num_im_list)

    @csutils.cache
    def is_track_index_by_net_z_location_consistent(self):
        track_i_dict = self.get_track_index_by_net_z_location()
        track_i_list = list(track_i_dict.values())

        if not track_i_list:
            _msg = (self._no_net_z_msg +
                    ' Cannot compute if track index is consistent across net '
                    'z locations. `track_i_list` is empty.')
            raise ValueError(_msg)

        return self.is_list_value_consistent(track_i_list)

    @csutils.cache
    def is_track_id_by_net_z_location_consistent(self):
        track_id_dict = self.get_track_id_by_net_z_location()
        track_id_list = list(track_id_dict.values())

        if not track_id_list:
            _msg = (self._no_net_z_msg +
                    ' Cannot compute if track id is consistent across net '
                    'z locations. `track_id_list` is empty.')
            raise ValueError(_msg)

        return self.is_list_value_consistent(track_id_list)

    @csutils.cache
    def is_net_z_location_spacing_consistent(
            self,
            *,
            sorting=MetadataSorting.ACQUISITION):

        z_loc_spacing = self.get_net_z_location_spacing(sorting=sorting)

        if not z_loc_spacing:
            msg = (self._no_seq_msg +
                   ' Cannot compute if z location spacing is consistent '
                   'because `z_loc_spacing` is empty.')
            raise ValueError(msg)

        return self.is_value_consistent(z_loc_spacing, is_numeric=True)

    @csutils.cache
    def is_net_z_location_by_sequence_consistent(self):
        z_loc_dict = self.get_net_z_location_by_sequence()
        z_loc_list = list(z_loc_dict.values())

        if not z_loc_list:
            _msg = (self._no_seq_msg +
                    ' Cannot compute if net z locations is consistent across '
                    'sequences. `z_loc_list` is empty.')
            raise ValueError(_msg)

        return self.is_list_value_consistent(z_loc_list)

    @csutils.cache
    def get_relative_time_spacing(self):
        return np.diff(self.starting_relative_time_coord_array).tolist()

    @csutils.cache
    def is_starting_relative_time_spacing_consistent(self):
        t_spacing = self.get_relative_time_spacing()
        if not t_spacing:
            msg = ('Cannot compute if time spacing is consistent because '
                   'no value for the time spacing can be computed.')
            raise ValueError(msg)

        return self.is_value_consistent(t_spacing)

    @csutils.cache
    def is_channel_index_by_frame_consistent(self):
        channel_index_dict = self.get_channel_index_by_frame()
        channel_index_list = list(channel_index_dict.values())

        if not channel_index_list:
            _msg = (self._no_frame_id_msg +
                    ' Cannot compute if channel index is consistent across '
                    'frames. `channel_index_list` is empty.')
            raise ValueError(_msg)

        return self.is_list_value_consistent(channel_index_list)

    @csutils.cache
    def is_xy_position_consistent(self, diff_threshold_um=0.5):
        return (self.is_value_consistent(self.x_center_position,
                                         is_numeric=True,
                                         diff_threshold=diff_threshold_um)
                and self.is_value_consistent(self.y_center_position,
                                             is_numeric=True,
                                             diff_threshold=diff_threshold_um))

    @csutils.cache
    def is_xy_size_consistent(self):
        return (self.is_value_consistent(self.x_size)
                and self.is_value_consistent(self.y_size))

    @csutils.cache
    def is_xy_pitch_consistent(self):
        return (self.is_value_consistent(self.x_pitch)
                and self.is_value_consistent(self.y_pitch))

    # getters
    # - simple properties
    @csutils.cached_property
    def dwell_time_us(self):
        return self.metadata.get_value(pvmk.DWELL_TIME)

    @csutils.cached_property
    def pockels(self):
        return self.metadata.get_value(self.POCKELS_KEY)

    @csutils.cached_property
    def has_tracks(self):
        return pvmk.TRACK_INDEX in self.metadata

    @csutils.cached_property
    def frame_id(self):
        return self.metadata.get_value(pvmk.FRAME_ID)

    @csutils.cached_property
    def two_d_image_id(self):
        return self.metadata.get_value(pvmk.TWO_D_IMAGE_ID)

    @csutils.cached_property
    def channel_index(self):
        return self.metadata.get_value(pvmk.CHANNEL_INDEX)

    @csutils.cached_property
    def channel_name(self):
        return self.metadata.get_value(pvmk.CHANNEL_NAME)

    @csutils.cached_property
    def filename(self):
        return self.metadata.get_value(pvmk.FILENAME)

    @csutils.cached_property
    def sequence_index(self):
        return self.metadata.get_value(pvmk.SEQUENCE_INDEX)

    @csutils.cached_property
    def relative_time(self):
        return self.metadata.get_value(pvmk.RELATIVE_TIME)

    def absolute_time(self):
        return self.metadata.get_value(pvmk.TIME)

    @csutils.cached_property
    def num_z_devices(self):
        return len(self.metadata[self.Z_POSITION_KEY])

    @csutils.cached_property
    def num_sequences(self):
        return len(self.sequence_index)

    @csutils.cached_property
    def num_frames(self):
        return sum(len(frame_id_list)
                   for frame_id_list in self.metadata[pvmk.FRAME_ID])

    @csutils.cached_property
    def z_location(self):
        z_loc_list = []
        for i_device in range(self.num_z_devices):
            device_key = str(i_device)
            z_loc = self.metadata.get_value(self.Z_POSITION_KEY + (device_key,))
            z_loc_list.append(z_loc)
        return tuple(z_loc_list)

    @csutils.cached_property
    def net_z_location(self):
        result = self.z_location[0]
        for loc in self.z_location[1:]:
            result = ArrayedMetadata.entry_binop(operator.add, result, loc)
        return result

    @csutils.cached_property
    def has_multiple_net_z_locations(self):
        net_z_loc = self.get_unique_net_z_locations()
        return isinstance(net_z_loc, list) and len(net_z_loc) > 1

    @csutils.cached_property
    def num_image_pixel_rows(self):
        return self.metadata.get_value(pvmk.LINES_PER_FRAME)

    @csutils.cached_property
    def y_size(self):
        return self.num_image_pixel_rows

    @csutils.cached_property
    def num_image_pixel_cols(self):
        return self.metadata.get_value(pvmk.PIXELS_PER_LINE)

    @csutils.cached_property
    def x_size(self):
        return self.num_image_pixel_cols

    @csutils.cached_property
    def x_pitch(self):
        return self.metadata.get_value((pvmk.MICRONS_PER_PIXEL, pvmk.X_AXIS))

    @csutils.cached_property
    def y_pitch(self):
        return self.metadata.get_value((pvmk.MICRONS_PER_PIXEL, pvmk.Y_AXIS))

    @csutils.cached_property
    def x_coord_array(self):
        return self.compute_coord_array(self.x_size,
                                        self.x_left_position,
                                        self.x_pitch)

    @csutils.cached_property
    def y_coord_array(self):
        return self.compute_coord_array(self.y_size,
                                        self.y_top_position,
                                        self.y_pitch)

    @csutils.cached_property
    def y_center_position(self):
        return self.metadata.get_value(self.Y_POSITION_KEY)

    @csutils.cached_property
    def y_top_position(self, auto_scalar=True, diff_threshold_um=0.5):
        if auto_scalar:
            center_pos = self.y_center_position
            if isinstance(center_pos, list):
                center_poss = np.array(center_pos)
                diff = center_poss.max() - center_poss.min()
                if diff > (2 * diff_threshold_um):
                    log.warning('y_center_position attribute is non-scalar and '
                                'is not consistent enough to convert to a'
                                'scalar.')
                else:
                    log.debug('Automatically converting non-scalar value'
                              'of y_center_position to scalar.')
                    center_pos = np.median(center_poss)

        return ArrayedMetadata.entry_nary_op(
            self.compute_leading_edge_pixel_position,
            center_pos,
            self.y_size,
            self.y_pitch)

    @csutils.cached_property
    def x_center_position(self):
        return self.metadata.get_value(self.X_POSITION_KEY)

    @csutils.cached_property
    def x_left_position(self, auto_scalar=True, diff_threshold_um=0.5):
        if auto_scalar:
            center_pos = self.x_center_position
            if isinstance(center_pos, list):
                center_poss = np.array(center_pos)
                diff = center_poss.max() - center_poss.min()
                if diff > (2 * diff_threshold_um):
                    log.warning('x_center_position attribute is non-scalar and '
                                'is not consistent enough to convert to a'
                                'scalar.')
                else:
                    log.debug('Automatically converting non-scalar value'
                              'of x_center_position to scalar.')
                    center_pos = np.median(center_poss)

        return ArrayedMetadata.entry_nary_op(
            self.compute_leading_edge_pixel_position,
            center_pos,
            self.x_size,
            self.x_pitch)

    @csutils.cached_property
    def z_size(self):
        return len(self.get_unique_net_z_locations())

    @csutils.cached_property
    def z_pitch(self):
        val_sorting = MetadataSorting.VALUE
        if not self.is_net_z_location_spacing_consistent(sorting=val_sorting):
            msg = ('Net z location spacing is not consistent; cannot compute '
                   'a value for `z_pitch`.')
            raise ValueError(msg)

        return self.get_net_z_location_spacing(sorting=val_sorting)[0]

    @csutils.cached_property
    def z_center_position(self):
        return (self.z_top_position + self.z_bottom_position) / 2

    @csutils.cached_property
    def z_top_position(self):
        z_locs = self.get_unique_net_z_locations(sorting=MetadataSorting.VALUE)
        return z_locs[0]

    @csutils.cached_property
    def z_bottom_position(self):
        z_locs = self.get_unique_net_z_locations(sorting=MetadataSorting.VALUE)
        return z_locs[-1]

    @csutils.cached_property
    def z_coord_array(self):
        return np.array(
            self.get_unique_net_z_locations(sorting=MetadataSorting.VALUE))

    @csutils.cached_property
    def starting_relative_time_coord_array(self):
        time_dict = self.get_starting_relative_time_by_sequence_index()
        out_list = []
        for i_seq in self.sequence_index:
            out_list.append(time_dict[i_seq])
        return np.array(out_list)

    @csutils.cached_property
    def time_size(self):
        return self.starting_relative_time_coord_array.size

    @csutils.cached_property
    def time_pitch(self):
        if not self.is_starting_relative_time_spacing_consistent():
            msg = ('Starting relative time spacing is not consistent; '
                   'cannot compute a value for `time_pitch`')
            raise ValueError(msg)
        return self.get_relative_time_spacing()[0]

    @csutils.cached_property
    def time_first_position(self):
        return self.starting_relative_time_coord_array[0]

    # class and static methods
    @staticmethod
    def get_entry_by_entry(by, get):
        a = by
        b = get
        zipped_entry_pair = ArrayedMetadata.entry_zip(a, b)
        zipped_entry_pair = ArrayedMetadata.linearize_entry(zipped_entry_pair)
        output_dict = dict()
        for aa, bb in zipped_entry_pair:
            try:
                result_list = output_dict[aa]
            except KeyError:
                output_dict[aa] = result_list = []
            result_list.append(bb)
        return output_dict

    @classmethod
    def is_list_value_consistent(cls, x, *, is_numeric=None):
        ref_len = len(x[0])
        ref_members = sorted(x[0])

        if is_numeric is None:
            is_numeric = (len(ref_members) > 0
                          and isinstance(ref_members[0], (float, int)))

        if is_numeric:
            ref_members = np.array(ref_members)

        for v in x:
            if not (len(v) == ref_len):
                return False

            if is_numeric:
                v_ndarray = np.array(sorted(v))
                if not (v_ndarray.shape == ref_members.shape):
                    return False
                arr_compare_crit = np.all(np.abs(v_ndarray - ref_members)
                                          < cls.DIFF_THRESHOLD)
            else:
                arr_compare_crit = sorted(v) == ref_members

            if not arr_compare_crit:
                return False

        return True

    @classmethod
    def is_value_consistent(cls, x, *, is_numeric=None, diff_threshold=None):
        if isinstance(x, list):
            if not x:
                return False

            if is_numeric is None:
                is_numeric = [isinstance(v, (float, int)) for v in x]

            if is_numeric:
                if diff_threshold is None:
                    diff_threshold = cls.DIFF_THRESHOLD

                x_ndarr = np.array(x)
                ref_val = x_ndarr[0]
                return bool(np.all(np.abs(x_ndarr - ref_val)
                                   < diff_threshold))
            else:
                return all(x[0] == xi for xi in x)

        elif isinstance(x, ScalarMetadata.ENTRY_TYPES):
            return True

        else:
            msg = (f'Unexpected type encountered when checking for value '
                   f'consistency; got {type(x).__name__}, expected one of '
                   f'{[t.__name__ for t in ArrayedMetadata.ENTRY_TYPES]}.')
            raise TypeError(msg)

    @staticmethod
    def list_dict_to_len_dict(d):
        return {k: len(v) for k, v in d.items()}

    @staticmethod
    def compute_leading_edge_pixel_position(center_pos, dim_size, dim_pitch):
        # FIXME - there might be a more robust way to compute this taking into
        #  account the ROI bounds
        return center_pos - (((dim_size - 1) / 2) * dim_pitch)

    @staticmethod
    def compute_coord_array(size, start, pitch):
        return start + (np.arange(size) * pitch)

    @staticmethod
    def filename_ref_deprecation_warning():
        log.warning('Do not use references to filename as an alias for unique'
                    ' 2d images; use the analogous 2d image methods.')


def load_dataset(*, index=None, config=None, dataset_info=None, **kwargs):
    if dataset_info is None:
        if config is None:
            config = get_global_config()
        else:
            config = ZMIAConfig(config)

        if index is None:
            msg = 'index must be specified if dataset_info is not passed.'
            raise RuntimeError(msg)
        dataset_info = config.dataset_list[index]
    else:
        if config is not None:
            log.warning('Config is being ignored since dataset_info was '
                        'passed.')
        if index is not None:
            log.warning('index is being ignored since dataset_info was '
                        'passed.')
            index = None

    config_name = dataset_info.config.name
    dataset_name = dataset_info.name

    dataset_str = dataset_name or '[unnamed dataset]'
    if index is not None:
        dataset_str = f'{dataset_str}, index={index}'

    config_str = f' {config_name}' if config_name else ''
    msg = 'Loading dataset ({}) from config{}.'
    log.info(msg, dataset_str, config_str)

    kwargs.setdefault('name', dataset_name)

    return PVDataset(dataset_info.full_path, **kwargs)


def load_all_datasets(config=None, do_fail=False, *args, **kwargs):
    """
    loads all the datasets from the configuration file

    :param config: (default is None, global config is used) dictionary of a
       loaded configuration file
    :param do_fail: (default is False) if True, errors will not be caught and
       will cause program to fail
    :param args:  passed to load_dataset and PVDataset initializer
    :param kwargs: passed to load_dataset and PVDataset initializer
    :return: a PVDatasetList of all dataset None will be appended if loading
       failed and do_fail parameter is set to True
    """
    if config is None:
        config = get_global_config()
    else:
        config = ZMIAConfig(config)

    datasets = PVDatasetList()
    for i_dataset, dataset_info in enumerate(config.dataset_list):
        try:
            d = load_dataset(dataset_info=dataset_info, *args, **kwargs)
        except Exception as e:
            d = None
            log.error('Loading of dataset {:d} failed.', i_dataset,
                      exc_info=e)
            if do_fail:
                raise

        datasets.append(d)

    if not datasets:
        log.warning('No datasets loaded, returning an empty list.')

    return datasets


def annotation_test():
    pass
#     config_path = os.path.join(
#         csutils.get_filepath(__file__),
#         '..',
#         'configs',
#         'default_config.yml')
#     set_global_config(config_path)
#     _config = get_global_config()
#
#     log = csutils.get_logger(
#         name='prairie_view_imports.annotation_test',
#         filepath=os.path.join(_config['log_directory'],
#                               'prairie_view_imports-annotation_test.log'))
#
#     dataset_idxs = [0, 2]
#
#     pvdatasets = [load_dataset(index=i, lazy=False) for i in dataset_idxs]
#
#     viewer = napari.Viewer(axis_labels=list(pvdatasets[0].dimensions))
#     layers = [viewer.add_image(data=pvd.I,
#                                scale=tuple(pvd.voxel_pitch_um),
#                                contrast_limits=[0, pvd.int_max],
#                                name=pvd.name)
#               for pvd in pvdatasets]
#
#     t_series_pvd = pvdatasets[dataset_idxs[0]]
#
#     region_labels_path = 'brain_labels'
#     try:
#         label_data, f = t_series_pvd.load_ndarray(region_labels_path)
#     except FileNotFoundError:
#         log.info('Creating blank label data.')
#         label_data = np.zeros(t_series_pvd.shape[-3:], dtype=int)
#     else:
#         files.append(f)
#         log.info(f'Loaded label data with {np.unique(label_data).size} labels.')
#
#     brain_region_labels = viewer.add_labels(
#         data=label_data,
#         scale=t_series_pvd.voxel_pitch_um,
#         name='brain_regions')
#
#     napari.run()
#
#     label_data = brain_region_labels.data
#     t_series_pvd.write_ndarray(label_data, region_labels_path)
#
#     log.info(f'Label data saved with {np.unique(label_data).size} labels.')
#
#     region_ids = np.unique(label_data)
#     region_ids = region_ids[region_ids != 0]
#     region_ids.sort()
#     n_regions = len(region_ids)
#
#     time_series = np.empty((n_regions, t_series_pvd.num_timepoints))
#     _bool_idx_arr = np.empty(t_series_pvd.shape, dtype=bool)
#     for i, _id in enumerate(region_ids):
#         _bool_idx_arr[()] = False
#         _bool_idx_arr[..., label_data == _id] = True
#         t_series_slice = t_series_pvd.I[_bool_idx_arr]
#         try:
#             t_series_slice.compute_chunk_sizes()
#         except AttributeError:
#             pass
#         time_series[i] = (t_series_slice
#                           .reshape((t_series_pvd.shape[0], -1))
#                           .mean(axis=1))
#
#     time_series_smooth = bn.move_mean(time_series,
#                                       window=6,
#                                       min_count=3,
#                                       axis=1)
#     time_series_baseline = np.percentile(time_series, 10, axis=1, keepdims=True)
#     time_series_delta_smooth = ((time_series_smooth - time_series_baseline)
#                                 / time_series_baseline)
#     time_series_delta = ((time_series - time_series_baseline)
#                          / time_series_baseline)
#     plt.plot(time_series_delta.T, alpha=0.3, lw=1)
#     plt.gca().set_prop_cycle(None)
#     lines = plt.plot(time_series_delta_smooth.T, lw=3)
#     for i, l in enumerate(lines):
#         l.set_label(f'Region {i + 1:02d}')
#     plt.legend()
#     plt.show()


def main(args):
    sources_path = os.path.join(os.path.split(__file__)[0])
    _file = csutils.no_ext_basename(__file__)

    if args.dev_test:
        csutils.apply_standard_logging_config(
            file_path=os.path.join(sources_path, '..', 'logs',
                                   f'{_file}.log'))
        annotation_test()
    else:
        if args.config_path is not None:
            config_path = args.config_path
            print(f'Attempting to load _config directly from "{config_path}"')
        else:
            if args.config is not None:
                config_name = args.config
            else:
                config_name = 'default_config'

            config_dir = os.path.join(csutils.get_filepath(__file__),
                                      '..',
                                      'configs')
            config_path = os.path.join(config_dir, config_name + '.yml')

            print(f'Attempting to load config "{config_name}" from '
                  f'config dir.')

        try:
            set_global_config(config_path)
        except Exception as e:
            print(f'Loading of configuration file failed. Quitting.')
            raise e

        try:
            log_dir = get_global_config()['log_directory']
        except KeyError:
            log_dir = './logs'

        csutils.touchdir(log_dir)
        log = csutils.get_logger(f'{__name__}.main')
        csutils.apply_standard_logging_config(
            window_level='debug' if args.verbose else 'info',
            window_format='cli',
            file_path=os.path.join(sources_path, '..', 'logs',
                                   f'{_file}.log'))

        log.info(f'Configuration file loaded from "{config_path}".')

        datasets = []

        if args.index is None:
            dataset_idx_iter = range(len(get_global_config()['datasets']))
        else:
            dataset_idx_iter = [args.index, ]

        for i in dataset_idx_iter:
            print()
            log.info('Attempting to load in dataset {:d} ...', i)
            try:
                dataset = load_dataset(index=i,
                                       lazy=args.lazy,
                                       ignore_cache=args.ignore_cache)
            except Exception as e:
                log.warning('Import of dataset {:d} failed with error:', i,
                            exc_info=e)
                datasets.append(None)
            else:
                datasets.append(dataset)

                log.info(f'Import of dataset {i:d} complete!')
                if args.display_attrs:
                    temp_name = datasets[-1].name
                    if temp_name is None:
                        temp_name = f'[Dataset {i:d}]'

                    # print_misc_dataset_attributes(datasets[-1], nme=temp_name)

        valid_datasets_enumerated = [
            (i, d) for i, d in zip(dataset_idx_iter, datasets)
            if d is not None]
        valid_datasets = [d for _, d in valid_datasets_enumerated]

        if not valid_datasets:
            log.warning('No datasets were imported successfully, exiting.')
            return

        for i, d in valid_datasets_enumerated:
            print()

            load_path = d._h5_path if d.directory is None else d.directory
            load_path = load_path.replace('\\', '\\\\')
            log.info('Load a PVDataset object for dataset {:d} by the '
                     'following command:', i)
            log.info('>>> prairie_view_imports.PVDataset(\'{}\')', load_path)

            if not args.no_save:
                save_path = d.save_to_numpy()
                save_path = save_path.replace('\\', '\\\\')
                log.info('Saved dataset {:d} to a numpy file. '
                         'Load by the following command:', i)
                log.info('>>> np.load(\'{}\')', save_path)
            else:
                log.info('Skipping saving of numpy data.')

        if args.view:
            print()
            log.info('Attempting to preview loaded image data...')
            try:
                axis_labels = []
                for d in valid_datasets:
                    if len(d.dimensions) > len(axis_labels):
                        axis_labels = d.dimensions

                viewer = napari.Viewer(axis_labels=axis_labels)

                layers = []
                for d in valid_datasets:
                    layers.append(
                        viewer.add_image(data=d.I,
                                         scale=tuple(d.voxel_pitch_um),
                                         contrast_limits=[0, d.int_max],
                                         name=d.name))

                log.info('Opening a napari session.')
                napari.run()
                log.info('Napari session ended.')
            except Exception as e:
                log.warning('Viewing of image data failed with error:',
                            exc_info=e)

        log.info('Import and conversion script successfully completed. '
                 'Quitting.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='import, convert, and view 2-photon image data obtained '
                    'from PrarieView imaging software.')

    parser.add_argument('config',
                        default=None,
                        nargs='?',
                        help='the name of the config file for the datasets '
                             'to be converted (default is "default_config") '
                             'file must be in the configs directory with '
                             'extension ".yml"')
    parser.add_argument('--ignore-cache',
                        action='store_true',
                        help='force load the data from .xml and raw '
                             'images; ignoring and overwriting any existing '
                             '.h5 files')
    parser.add_argument('--config-path', '-p',
                        default=None,
                        help='explicit path to a config file; `config` '
                             'argument will be ignored')
    parser.add_argument('--index', '-i',
                        default=None,
                        type=int,
                        help='index (zero-indexed) of the dataset for '
                             'processing, if this flag is not passed, all '
                             'datasets in the config file will be converted.')
    parser.add_argument('--view',
                        action='store_true',
                        help='flag to open a napari session to '
                             'view the datasets; datasets will attempt to be '
                             'overlayed by using PrairieView metadata.')
    parser.add_argument('--lazy', '-l',
                        action='store_true',
                        help='attempt only read in image data as it is needed,'
                             'rather than importing all files at once. '
                             '*IN DEVELOPMENT*')
    parser.add_argument('--display-attrs', '-a',
                        action='store_true',
                        help='display misc attributes of each dataset after '
                             'loading.')
    parser.add_argument('--no-save',
                        action='store_true',
                        help='do not save the imported data to a .npy file.')
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='display all messages logged to the console.')
    parser.add_argument('--dev-test',
                        action='store_true',
                        help='ignore all arguments and run development/'
                             'testing scripts.')

    _args = parser.parse_args()

    try:
        main(_args)
    finally:
        close_msg = 'Closing all files opened in "{}".'
        log.debug(close_msg, __name__)

        for f in files:
            try:
                f.close()
            finally:
                continue

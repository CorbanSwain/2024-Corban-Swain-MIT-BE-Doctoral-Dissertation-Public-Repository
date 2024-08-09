#!python3
# utilities.py

# imports
from __future__ import annotations

import copy
import enum
import fractions
import os
from collections import namedtuple, UserDict
import numpy as np
import h5py
import time
import nd2reader
import contextlib
from collections.abc import Generator, Mapping
from typing import Tuple, Union, Optional, Callable, Protocol, List, Any
from types import ModuleType
import skimage as ski
import skimage.io

# - local imports
import c_swain_python_utils as csutils

__all__ = [
    # functions
    'get_global_config',
    'set_global_config',
    'choose_dataset_loader',
    'imread',
    # classes
    'ZMIAConfig',
    'ZMIADatasetInfo',
    'Criterion',
    'CriteriaList',
    'NISDataset',
    'DatasetLoader',
    'ZmiaDatasetInfoKeys',
    'ZmiaConfigKeys',
    'Dimension',
    'Dim',
    # constants
    'CACHE_DIR_NAME',
    'PICKLE_EXTENSION',
    'XML_EXTENSION',
    'NUMPY_EXTENSION',
    'H5_EXTENSION',
    'TIFF_EXTENSION'
]


# for napari viewer
os.environ.update({'NUMEXPR_MAX_THREADS': str(os.cpu_count())})

# constants
H5_PATH_SEP = '/'
H5PY_LIBVER = 'latest'

CACHE_DIR_NAME = '.py_caches'

H5_EXTENSION = '.h5'
PICKLE_EXTENSION = '.pkl'
XML_EXTENSION = '.xml'
NUMPY_EXTENSION = '.npy'
TIFF_EXTENSION = '.tif'

# - package config constants
# TODO - add these to .yaml file specific to the package setup
SKIMAGE_IO_PLUGIN = 'pil'


log = csutils.get_logger(__name__)

# configurations
_global_config: Optional[ZMIAConfig] = None


def get_global_config() -> ZMIAConfig:
    global _global_config
    if _global_config is None:
        msg = 'No global configuration has been set/loaded.'
        raise RuntimeError(msg)
    return _global_config


def set_global_config(config: os.PathLike | ZMIAConfig | dict
                      ) -> None:
    global _global_config
    _global_config = ZMIAConfig(config)


class Dimension(str, enum.Enum):
    X = 'x'
    Y = 'y'
    CHANNEL = 'channel'
    Z = 'z'
    POSITION = 'position'
    TIME = 'time'

    @property
    def nd2_str(self):
        if self is Dimension.X:
            return 'x'
        elif self is Dimension.Y:
            return 'y'
        elif self is Dimension.CHANNEL:
            return 'c'
        elif self is Dimension.Z:
            return 'z'
        elif self is Dimension.POSITION:
            return 'v'
        elif self is Dimension.TIME:
            return 't'

    @property
    def is_numeric(self):
        return self in (Dim.X, Dim.Y, Dim.Z, Dim.TIME)

    def __repr__(self):
        return f'Dim(\'{self:s}\')'


Dim = Dimension  # convenience renaming


class _UserDict(UserDict):
    # protected methods
    def _getitem_or_none(self, key):
        try:
            return self.data[key]
        except KeyError:
            return None


class _ImmutableUserDict(_UserDict):
    def __setitem__(self, *args):
        self._fail_on_modification()

    def __delitem__(self, *args):
        self._fail_on_modification()

    def _fail_on_modification(self):
        msg = f'{self.__class__.__name__}\'s dictionary cannot be modified.'
        raise RuntimeError(msg)


def imread(*args, plugin=SKIMAGE_IO_PLUGIN, **kwargs):
    """
    wrapper for skimage.io.imread
    """
    return ski.io.imread(*args, plugin=SKIMAGE_IO_PLUGIN, **kwargs)


class ZmiaConfigKeys:
    PATH = 'path'
    NAME = 'name'
    DESCRIPTION = 'description'
    DATA_DIR = 'data_directory'
    LOG_DIR = 'log_directory'
    DATASET_LIST = 'datasets'
    OUTPUT_DIR = 'output_directory'
    DEFAULTS = 'defaults'
    SET_DEFAULTS = 'set-defaults'
    RESET_DEFAULTS = 'reset-defaults'


zck = ZmiaConfigKeys


class ZMIAConfig(_ImmutableUserDict):
    """
    class for reading from a zmia configuration yaml file
    """

    lookup_dict = dict()

    NAME_KEY = 'name'
    DATA_DIR_KEY = 'data_directory'
    LOG_DIR_KEY = 'log_directory'
    DATASET_LIST_KEY = 'datasets'
    OUTPUT_DIRECTORY_KEY = 'output_directory'
    DEFAULTS_KEY = 'defaults'

    # magic methods
    def __new__(cls,
                config_like: str | dict | ZMIAConfig,
                /):
        if isinstance(config_like, str):
            try:
                return cls.lookup_dict[config_like]
            except KeyError:
                pass
        elif isinstance(config_like, cls):
            return config_like
        return super().__new__(cls)

    def __init__(self,
                 config_like: str | dict,
                 /):
        super().__init__()

        if isinstance(config_like, str):
            self.data = csutils.read_yaml(config_like)
            self.path = config_like
        elif isinstance(config_like, dict):
            self.data = config_like
            self.path = None
        else:
            msg = (f'Cannot create config from passed argument, config_like='
                   f'{repr(config_like)}.')
            raise ValueError(msg)

        if self.name:
            ZMIAConfig.lookup_dict[self.name] = self

        if self.output_directory is not None:
            csutils.touchdir(self.output_directory)

        if not os.path.exists(self.data_directory):
            log.warning('Input data directory in configuration file does not '
                        'exist\n"{}".', self.data_directory)

    def __getitem__(self, item):
        if item == zck.DATASET_LIST:
            log.warning('It is not recommended to access the `{}` entry '
                        'directly, use the `{}.dataset_list` property instead.',
                        zck.DATASET_LIST,
                        self.__class__.__name__)
        return super().__getitem__(item)

    def __hash__(self):
        return hash(self.path)

    # general methods
    def get_dataset_info(self, *, index=None, **kwargs) -> ZMIADatasetInfo:
        if index is not None and len(kwargs) > 0:
            log.warning('index parameter was passed to get_dataset; ignoring '
                        'all other filter parameters.')

        if index is not None:
            return self.dataset_list[index]
        else:
            out_list = []
            for d in self.dataset_list:
                do_include = True
                for kw, v in kwargs.items():
                    if (kw not in d) or (d[kw] != v):
                        do_include = False
                        break

                if do_include:
                    out_list.append(d)

            if len(out_list) > 1:
                msg = (f'Multiple ({len(out_list)}) matching datasets found, '
                       f'filter criterion should be more strict.')
                raise ValueError(msg)
            elif len(out_list) == 0:
                msg = f'No matching datasets found with filter criterion, ' \
                      f'{kwargs} .'
                raise ValueError(msg)
            else:
                return out_list[0]

    def get_dataset(self, *args, **kwargs):
        log.warning('{}.get_dataset() is deprecated, use get_dataset_info()'
                    'instead.', ZMIAConfig.__name__)
        return self.get_dataset_info(*args, **kwargs)

    # getters
    @property
    def name(self):
        return self._getitem_or_none(zck.NAME)

    @property
    def data_directory(self):
        return self._getitem_or_none(zck.DATA_DIR)

    @property
    def log_directory(self):
        return self._getitem_or_none(zck.LOG_DIR)

    @property
    def _defaults(self):
        defaults_dict = self._getitem_or_none(zck.DEFAULTS)
        if defaults_dict is None:
            return dict()

        return defaults_dict

    @property
    def dataset_list(self) -> Optional[List[ZMIADatasetInfo]]:
        dataset_list = self._getitem_or_none(zck.DATASET_LIST)
        if dataset_list is None:
            return dataset_list

        global_defaults = self._defaults
        running_defaults = copy.deepcopy(global_defaults)

        output_dict_list = []

        ele: dict
        for ele in dataset_list:
            ele_keys = list(ele.keys())
            if len(ele_keys) == 1:
                ele_key = ele_keys[0]
                ele_key_lower = ele_key.lower()
                if zck.SET_DEFAULTS == ele_key_lower:
                    running_defaults.update(ele[ele_key])
                    continue
                if zck.RESET_DEFAULTS == ele_key_lower:
                    running_defaults = copy.deepcopy(global_defaults)
                    continue

            new_dict = copy.deepcopy(running_defaults)

            new_dict.update(ele)

            format_string_entries(new_dict, format_lookup=new_dict)

            if zck.SET_DEFAULTS in new_dict or zck.RESET_DEFAULTS in new_dict:
                log.warning('Encountered defaults flag in a dataset entry: {}',
                            new_dict)

            output_dict_list.append(new_dict)

        return [ZMIADatasetInfo(d, self) for d in output_dict_list]

    @property
    def output_directory(self) -> Optional[str]:
        return self._getitem_or_none(zck.OUTPUT_DIR)


def format_string_entries(dict_to_format: dict,
                          format_lookup: dict
                          ) -> None:

    """
    updates all string entries in the dict_to_format using the format_lookup
    dictionary. Format is flagged similar to `.format` formatting except double
    curly braces indicate capture clauses.

    :param dict_to_format:
    :param format_lookup:
    :return:
    """
    for k, v in dict_to_format.items():
        if isinstance(v, (dict, Mapping)):
            format_string_entries(dict_to_format[k], format_lookup)
            continue

        if not isinstance(v, str):
            continue

        val_str: str = v
        formatted_str = val_str.format(**format_lookup)
        dict_to_format[k] = formatted_str


class ZmiaDatasetInfoKeys:
    TYPE = 'type'
    PATH = 'path'
    NAME = 'name'
    DESCRIPTION = 'description'
    XY_PITCH_UM = 'xy-pitch-um'
    Z_PITCH_UM = 'z-pitch-um'
    RUN_ID = 'run-id'
    RUN_ID_DEPRECATED = 'run'


zdik = ZmiaDatasetInfoKeys


class ZMIADatasetInfo(_ImmutableUserDict):

    NAME_KEY = 'name'
    PATH_KEY = 'path'
    TYPE_KEY = 'type'
    DESCRIPTION_KEY = 'description'

    def __init__(self, info_dict, config):
        super().__init__()
        self.data: dict = info_dict
        self.config: ZMIAConfig = config

        if self.PATH_KEY not in self.data:
            log.warning('{} does not have an associated path.',
                        self)
        elif not os.path.exists(self.full_path):
            log.warning('{} ''s full path does not exist.\n\t"{}"',
                        self, self.full_path)

    def __hash__(self):
        return hash(f'{self.config.path},{self.full_path},'
                    f'{self.type or "None"}')

    @property
    def name(self) -> Optional[str]:
        dataset_name: Optional[str] = self._getitem_or_none(zdik.NAME)
        dataset_type: Optional[str] = self.type
        if dataset_name is None:
            if dataset_type is None:
                return None
            else:
                return f'{dataset_type} : [unnamed]'
        else:
            if dataset_type is None:
                return dataset_name
            else:
                return f'{dataset_type}-{dataset_name}'

    @property
    def path(self) -> str:
        return self.data[zdik.PATH]

    @property
    def full_path(self) -> str:
        return os.path.join(self.config.data_directory, self.path)

    @property
    def type(self) -> Optional[str]:
        return self._getitem_or_none(zdik.TYPE)

    @property
    def description(self) -> Optional[str]:
        return self._getitem_or_none(zdik.DESCRIPTION)

    @property
    def file_extension(self) -> str:
        return os.path.splitext(self.path)[1]

    @property
    def run_id(self):
        out = self._getitem_or_none(zdik.RUN_ID)
        if out is None:
            out = self._getitem_or_none(zdik.RUN_ID_DEPRECATED)
        return out


# criteria-checking helper classes
_Criterion = namedtuple('_Criterion', ('criterion', 'message'))


class Criterion(_Criterion):
    def __bool__(self):
        return self.criterion


class CriteriaList(object):
    def __init__(self, crit_list):
        self._crit_list = []
        self.crit_list = crit_list

    def __bool__(self):
        return bool(all(self.crit_list))

    def check_crits_and_raise(self, *, message_prefix=''):
        for crit in self.crit_list:
            if not crit:
                raise ValueError(message_prefix + crit.message)

    def check_crits_and_log(self, *, message_prefix='', log_fcn=log.warning):
        for crit in self.crit_list:
            if not crit:
                msg = message_prefix + crit.message
                log_fcn(msg)

    @property
    def crit_list(self):
        return self._crit_list

    @crit_list.setter
    def crit_list(self, x):
        assert isinstance(x, list), '`crit_list` must be a list object.'
        assert all(isinstance(i, Criterion) for i in x), (
            'All elements of `crit_list` must be Criterion objects.')
        self._crit_list = x


def load_swmr_h5(filepath, writer_file_optional=None, mode='r', *args,
                 **kwargs):
    if writer_file_optional is None or not bool(writer_file_optional):
        if mode == 'r':
            initial_open_mode = 'r+'
        else:
            initial_open_mode = mode
        writer_file = h5py.File(filepath,
                                initial_open_mode,
                                libver=H5PY_LIBVER,
                                *args, **kwargs)
        writer_file.swmr_mode = True
    else:
        writer_file = writer_file_optional

    if mode == 'r':
        reader_file = h5py.File(filepath, mode, libver=H5PY_LIBVER, *args,
                                **kwargs)
        return reader_file, writer_file
    else:
        if mode != writer_file.mode:
            msg = f'To maintain compatibility with SWMR mode, `h5py.File` ' \
                  f'will be returned in "{writer_file.mode}" mode.'
            log.debug(msg) if log else print(msg)
        return writer_file, writer_file


def dict_to_h5_group(d: dict, grp: h5py.Group):
    for k, v in d.items():
        if log:
            log.debug(f'...writing dict entry {{{k}: {v}}} to h5 group '
                    f'"{grp.name}."')

        k_str = str(k)
        if isinstance(v, dict):
            subgrp = grp.create_group(k_str)
            dict_to_h5_group(v, subgrp)
        else:
            grp[k_str] = v


class NISDataset(object):

    HEIGHT_KEY = 'height'
    WIDTH_KEY = 'width'
    PIXEL_PITCH_KEY = 'pixel_microns'
    Z_COORD_KEY = 'z_coordinates'
    Z_LEVELS_KEY = 'z_levels'
    CHANNELS_KEY = 'channels'

    def __init__(self,
                 path,
                 name=None,
                 lazy=True):
        self.path = path
        self.name = name
        self._image_ndarray = None

        with nd2reader.ND2Reader(self.path) as nd2_images:
            self.metadata = nd2_images.metadata
            self.sizes = nd2_images.sizes

        if not lazy:
            self.load_all_image_data()

    # methods
    def load_all_image_data(self, force=False):
        if (not force) and (self._image_ndarray is not None):
            log.debug('Images already loaded for NISDataset.')
            return

        log.info('Loading all images from ND2 file, this may take some '
                 'time.')

        with nd2reader.ND2Reader(self.path) as nd2_images:
            nd2_images.iter_axes = self.non_page_axes
            # log.debug('{:s}', str(nd2_images.metadata))
            self._image_ndarray = np.array(nd2_images)

        new_shape = []
        for ax in self.non_page_axes:
            new_shape.append(self.sizes[ax])
        new_shape = tuple(new_shape)
        new_shape = new_shape + self.page_shape

        self._image_ndarray = np.reshape(self._image_ndarray,
                                         new_shape)


    @property
    def num_images(self):
        n = 1
        for k, v in self.sizes:
            if k in ('x', 'y'):
                continue
            n *= v
        return n

    @property
    def page_shape(self):
        return (self.metadata[self.HEIGHT_KEY],
                self.metadata[self.WIDTH_KEY])

    @property
    def non_page_axes(self):
        non_page_ax = []
        for ax in self.default_axis_order():
            if (ax not in ('y', 'x')
                    and ax in self.sizes
                    and self.sizes[ax] > 1):
                non_page_ax.append(ax)
        return non_page_ax

    @property
    def z_locs(self):
        z_coords = self.metadata[self.Z_COORD_KEY]
        return [z_coords[i]
                for i in self.metadata[self.Z_LEVELS_KEY]]

    @property
    def pixel_pitch_um(self):
        return self.metadata[self.PIXEL_PITCH_KEY]

    @property
    def image_ndarray(self):
        self.load_all_image_data()
        return self._image_ndarray

    @property
    def channel_name_list(self):
        return self.metadata[self.CHANNELS_KEY]

    # class methods
    @classmethod
    def from_dataset_info(cls, *, dataset_info, **kwargs):
        nd2_path = dataset_info.full_path
        kwargs.setdefault('name', dataset_info.name)
        return cls(nd2_path, **kwargs)

    # static method
    @staticmethod
    def default_axis_order():
        import imaging_dataset as imd
        return [d.nd2_str for d in imd.default_dimension_order]


class DatasetLoader(Protocol):
    def __call__(self,
                 *args,
                 dataset_info: ZMIADatasetInfo,
                 **kwargs
                 ) -> Any: ...


def choose_dataset_loader(*,
                          dataset_type: Optional[str] = None,
                          dataset_info: Optional[ZMIADatasetInfo] = None,
                          ) -> DatasetLoader:
    """
    determine the function for loading the dataset based on the dataset type
    """
    import prairie_view_imports as pvi
    from tiff_utils import TIFFDataset
    from video_utils import AVDataset

    raw_file_extension: Optional[str] = None
    if dataset_info is not None:
        if dataset_type is not None:
            log.warning('Defaulting to use the dataset type specified in the '
                        '`dataset_info` object; ignoring the `dataset_type` '
                        'str. Pass one parameter or the other (not both) to '
                        'suppress this message.')

        raw_file_extension = dataset_info.file_extension

    if dataset_type is None:
        if dataset_info is None:
            msg = (f'`dataset_type` str or `dataset_info` object must be '
                   f'passed; received neither.')
            raise RuntimeError(msg)

        dataset_type = dataset_info.type

    if dataset_type == 'two-photon':
        return pvi.load_dataset
    elif dataset_type == 'behavioral-camera':
        return AVDataset.from_dataset_info
    elif dataset_type == 'confocal':
        if not raw_file_extension:
            return NISDataset.from_dataset_info

        if raw_file_extension.lower() in ['.tiff', '.tif']:
            return TIFFDataset.from_dataset_info
        else:
            return NISDataset.from_dataset_info
    else:
        msg = (f'Provided dataset type does not have an associated reader, '
               f'"{dataset_type}".')
        raise ValueError(msg)
#!python3
# imaging_dataset.py

from __future__ import annotations
import pickle
from typing import Union, Optional, Callable, Any, Iterable
import dask.array
import os
import copy
import numpy as np
import enum
import skimage.io
from collections import UserDict
import time

# - local imports
from utilities import *
import prairie_view_imports as pvi
import c_swain_python_utils as csutils
from tiff_utils import TIFFDataset
from video_utils import AVIDataset, AVDataset

COORDINATES_KEY = 'coordinates'

# logging
log = csutils.get_logger(__name__)


class ImagingDatasetDimensionality(enum.Flag):
    SINGLE_IMAGE = 0
    MULTI_CHANNEL = enum.auto()
    MULTI_Z = enum.auto()
    MULTI_POSITION = enum.auto()
    MULTI_TIME = enum.auto()


ImDDim = ImagingDatasetDimensionality  # convenience renaming

default_dimension_order = (
    Dim.TIME,
    Dim.POSITION,
    Dim.Z,
    Dim.CHANNEL,
    Dim.Y,
    Dim.X)
default_dim_order_dict = dict(zip(default_dimension_order,
                                  range(len(default_dimension_order))))
default_page_dimension_order = default_dimension_order[-2:]
default_non_page_dimension_order = default_dimension_order[:-2]

PAGE_NDIMS = 2

nd2_str_to_dim = dict(zip(NISDataset.default_axis_order(),
                          default_dimension_order))


SCALE_PLACEHOLDER = 1


class CoordinateVector:
    # magic methods
    def __init__(self,
                 vector: np.ndarray,
                 unit: Optional[str] = None):
        self.vector: np.ndarray = vector
        self.unit: Optional[str] = unit

    def __eq__(self, other: CoordinateVector):
        if not isinstance(other, CoordinateVector):
            return False

        if not (len(self.vector) == len(other.vector)):
            return False

        if not (self.unit == other.unit):
            return False

        return all(si == oi for si, oi in zip(self.vector, other.vector))

    def __repr__(self):
        return (f'Coord([{self.vector[0]}, ...], ' 
                f'len={len(self.vector):d}, unit=\'{self.unit}\')')

    # properties
    @property
    def median_pitch(self):
        return np.median(np.diff(self.vector))

    # general methods
    def flip(self, in_place: bool = False):
        if in_place:
            self.vector = np.flip(self.vector, axis=0)
        else:
            new_cv = self.__class__(np.flip(self.vector, axis=0),
                                    unit=self.unit)
            return new_cv


class ImagingDataset(object):
    """
    generalized class for working with imaging data
    """
    # magic methods
    def __init__(
            self,
            *,
            path:                 str | bytes | os.PathLike,
            image_ndarray:        Optional[np.ndarray] = None,
            image_daskarray:      Optional[dask.array.Array] = None,
            get_image_data_func:  Optional[Callable] = None,
            dimensions:           Optional[tuple[Dim, ...]] = None,
            coordinates:          Optional[dict[Dim, CoordinateVector]] = None,
            name:                 Optional[str] = None,
            position:             Optional[dict[Dim, float]] = None,
            source_data:          Optional[Any] = None,
            raw_data:             Optional[Any] = None,
            id_str:               Optional[str] = None,
            cache_path:           Optional[str | bytes | os.PathLike] = None,
            op_history:           Optional[list[str]] = None,
            use_cache:            bool = True,
            clear_cache:          bool = False,
            clear_cache_and_exit: bool = False,
            cache_after_init:     bool = False,
            check_for_image_data: bool = True):
        """General Imaging Dataset

        :param str | bytes | os.PathLike path:
            path where this dataset should be stored, if the path is a directory
            it will be where cache files and other outputs are placed by default
            If a path to a file is passed, the directory containing that file
            will fill the same purpose.

        :param image_ndarray:
        :param image_daskarray:
        :param get_image_data_func:
        :param dimensions:
        :param coordinates:
        :param name:
        :param position:
        :param source_data:
        :param Any | None raw_data:
            Data or dataset object used as the source for this ImagingDataset
            instance

            .. deprecated:: 0.1
                Use ``source_data`` parameter instead

        :param id_str:
        :param cache_path:
        :param use_cache:
        :param clear_cache:
        :param clear_cache_and_exit:
        :param cache_after_init:
        :param check_for_image_data:
        """
        super().__init__()

        self.name = name
        self.path = path
        self._id = id_str
        self.op_history = op_history

        if source_data is None:
            self.source_data = raw_data
        else:
            self.source_data = source_data

        if clear_cache or clear_cache_and_exit:
            self.clear_cache(cache_path)
            if clear_cache_and_exit:
                log.info('Exiting init function after clearing cache.')
                return

        self.dimensions = dimensions
        self.coordinates = coordinates
        self.position = position

        if raw_data is not None:
            log.warning('Use of the `raw_data` init parameter is deprecated '
                        'use `source_data` instead.')

        self._image_ndarray = None
        self._image_daskarray = None
        self._get_image_data_func = None

        # FIXME - this should not be an instance attribute (at least not with
        #  this implementation), maybe a direct inspection self._image_ndarray
        #  and self._image_daskarray via a computed property or
        #  something ... not sure. Will require a refactor, this attribute
        #  is also referenced in instance method self._init_from_cache() and
        #  class method ImagingDataset.from_pvdataset().
        self._did_initialize_image_data = False

        if use_cache:
            self._init_from_cache(cache_path=cache_path)

        if self._did_initialize_image_data:
            # successfully loaded data from cache
            if (image_ndarray is not None) or (image_daskarray is not None):
                log.warning('Ignoring the explicitly passed image data and '
                            'using image data from cache file instead. Either '
                            'set the `use_cache` flag to False or do not pass '
                            'image data to suppress this warning.')
        else:
            self._get_image_data_func = get_image_data_func
            log.debug('self._get_image_data_func = {}',
                      repr(get_image_data_func))

            if image_ndarray is not None:
                self._image_ndarray = image_ndarray
                self._did_initialize_image_data = True

            if image_daskarray is not None:
                self._image_daskarray = image_daskarray
                self._did_initialize_image_data = True

            if ((self._did_initialize_image_data
                    or (self._get_image_data_func is not None))
                    and cache_after_init):
                self.cache_to_disk()

        if (check_for_image_data and (not self._did_initialize_image_data) and
                (self._get_image_data_func is None)):
            log.warning('No image data was loaded nor was a loading '
                        'function passed when initializing this {} object.',
                        self.__class__.__name__)

    def __repr__(self):
        name = '(None)' if self.name is None else f'\'{self.name}\''
        return (f'{self.__class__.__name__}(name={name}, '
                f'shape={tuple(self.shape)})')

    def drop_slices(
            self,
            slice_indexes: list[int],
            axis: Dimension | int,
            in_place: bool = False,
            **kwargs) -> ImagingDataset | None:

        if in_place:
            raise NotImplementedError('in-place modification not yet '
                                      'implemented.')

        slice_dim: Dimension
        if isinstance(axis, Dimension):
            slice_dim = axis
        else:
            # axis is an instance of int
            slice_dim = self.dimensions[axis]

        kept_slices = [i for i in range(self.shape_dict[slice_dim])
                       if (i not in slice_indexes)]
        kept_slices_bool = np.zeros((self.shape_dict[slice_dim], ),
                                    dtype=bool)
        kept_slices_bool[kept_slices] = True

        full_index = []
        for dim in self.dimensions:
            if dim is slice_dim:
                full_index.append(kept_slices)
                continue
            full_index.append(slice(None))

        full_index = tuple(full_index)

        new_obj = self.apply(np.ndarray.__getitem__,
                             op_args=[full_index, ],
                             in_place=in_place,
                             **kwargs)

        coord_vector = new_obj.coordinates[slice_dim].vector
        new_obj.coordinates[slice_dim] = CoordinateVector(
            vector=coord_vector[kept_slices_bool],
            unit=new_obj.coordinates[slice_dim].unit)

        return new_obj

    def rot90(self,
              axis: Dimension | int = Dimension.Z,
              *,
              k: int = 1,
              in_place: bool = False) -> ImagingDataset | None:

        if in_place:
            raise NotImplementedError('in-place modification not yet '
                                      'implemented.')

        rotation_axis_dim: Dimension
        if isinstance(axis, Dimension):
            rotation_axis_dim = axis
        else:
            rotation_axis_dim = self.dimensions[axis]

        if rotation_axis_dim not in (Dim.X, Dim.Y, Dim.Z):
            msg = ('Rotation about a non spatial dimension is not defined; '
                   'pass a dimension corresponding to X, Y or Z for rot90.')
            raise ValueError(msg)

        rotation_plane_axes: tuple[int, int]
        try:
            if rotation_axis_dim is Dim.X:
                rotation_plane_axes = (
                    self.axis_dict[Dim.Z], self.axis_dict[Dim.Y])
            elif rotation_axis_dim is Dim.Y:
                rotation_plane_axes = (
                    self.axis_dict[Dim.X], self.axis_dict[Dim.Z])
            else: # Z rotation
                rotation_plane_axes = (
                    self.axis_dict[Dim.Y], self.axis_dict[Dim.X])
        except KeyError as e:
            msg = (f'Imaging data does not have sufficient spatial dimensions '
                   f'to apply a rotation about the requested axis. '
                   f'({rotation_axis_dim = }; {self.shape_dict = })')
            raise RuntimeError(msg) from e

        new_obj = self.apply(np.rot90,
                             in_place=in_place,
                             op_kwargs=dict(k=k,
                                            axes=rotation_plane_axes))

        dim_0, dim_1 = (new_obj.dimensions[i] for i in rotation_plane_axes)
        for _ in range(k % 4):
            coord_0 = new_obj.coordinates[dim_0]
            coord_1 = new_obj.coordinates[dim_1]
            new_obj.coordinates[dim_0] = coord_1.flip()
            new_obj.coordinates[dim_1] = coord_0

        log.warning('position data attribute may be out of date after '
                    'rotation (proper handling not yet implemented).')

        return new_obj

    def flip(self,
             axis: Dim | int,
             *,
             flip_coordinates: bool = False,
             in_place: bool = False) -> ImagingDataset | None:

        if in_place:
            raise NotImplementedError('in-place modification not yet '
                                      'implemented.')

        new_obj = self.apply(
            np.flip,
            axis=axis,
            in_place=in_place)

        if flip_coordinates:
            flip_dim: Dimension
            if isinstance(axis, Dimension):
                flip_dim = axis
            else:
                # axis is an instance of int
                flip_dim = self.dimensions[axis]

            new_obj.coordinates[flip_dim] = (
                new_obj.coordinates[flip_dim].flip())

        return new_obj

    def apply(
            self,
            op: Callable[[Any, ...], np.ndarray | dask.array.Array],
            axis: Optional[Dim | int] = None,
            *,
            in_place: bool = False,
            op_args: Optional[Iterable[any]] = None,
            op_kwargs: Optional[dict[str, any]] = None,
            use_cache: bool = False
    ) -> ImagingDataset | None:

        _op_name: str = op.__name__
        _op_module: Optional[str]
        try:
            _op_module = op.__module__
        except AttributeError:
            _op_module = None

        _op_class: Optional[str]
        try:
            _op_class = op.__class__.__name__
        except AttributeError:
            _op_class = None

        axis_index: Optional[int]
        if axis is None:
            axis_index = None
        elif isinstance(axis, Dim):
            axis_index = self.axis_dict[axis]
        else:
            axis_index = axis

        op_args = op_args or tuple()
        op_kwargs = op_kwargs or dict()
        if ('axis' in op_kwargs) and (axis_index is not None):
            log.warning('Ignoring the keyword value passed for `axis` and '
                        'defaulting to use the positional argument\'s value; '
                        'pass only one of these arguments to suppress this  '
                        'warning.')

        if axis_index is not None:
            op_kwargs['axis'] = axis_index
        elif 'axis' in op_kwargs:
            axis_index = op_kwargs['axis']

        op_str = (_op_module + '.') if _op_module else ''
        op_str += f'{_op_name}(*{op_args}, **{op_kwargs})'

        log.info('Performing the following operation on {!s} {}:',
                 self,
                 ('\"in place\"' if in_place
                  else 'and returning a new ImagingDataset'))
        log.info('> ' + op_str)

        result: np.ndarray = (
            op(self.get_image_data(), *op_args, **op_kwargs))

        did_reduce_dim: bool = result.ndim == (self.ndim - 1)
        did_keep_dim: bool = result.ndim == self.ndim

        new_dimensions: Optional[tuple[Dim, ...]] = None
        new_coordinates: Optional[dict[Dim, CoordinateVector]] = None
        new_position: Optional[dict[Dim, float]] = None

        if did_keep_dim:
            new_dimensions = copy.deepcopy(self.dimensions)
            new_coordinates = copy.deepcopy(self.coordinates)
            new_position = copy.deepcopy(self.position)
        elif did_reduce_dim and (axis_index is not None):
            if self.dimensions:
                new_dimensions = tuple([d for i, d in enumerate(self.dimensions)
                                        if i != axis_index])
            if self.coordinates:
                new_coordinates = {dim: coord for dim, coord
                                   in self.coordinates.items()
                                   if axis_index != self.axis_dict[dim]}
            if self.position:
                new_position = {dim: pos for dim, pos
                                in self.position.items()
                                if axis_index != self.axis_dict[dim]}
        else:
            log.warning('Encountered an unexpected dimensionality change '
                        'as a result of the operation `{:s}`. {} attributes '
                        'will not be set for the created {} instance.'
                        'starting shape: {}, resulting shape: {}',
                        op_str, ['dimensions', 'coordinates', 'position'],
                        self.__class__.__name__, self.shape_dict, result.shape)

        new_history = (copy.deepcopy(self.op_history) or list())
        new_history.append(op_str)

        new_name = (f'{_op_name.upper()}({abs(hash(self.name + op_str)):x})'
                    f'-{self.name}')

        if in_place:
            raise NotImplementedError(
                'in-place function application not yet implemented.')

        return self.__class__(
            path=copy.deepcopy(self.path),
            name=new_name,
            dimensions=new_dimensions,
            coordinates=new_coordinates,
            op_history=new_history,
            position=new_position,
            image_ndarray=result,
            source_data=self,
            use_cache=use_cache)

    # general methods
    def clear_cache(self, cache_path=None, memory=True, disk=True):
        if memory:
            self._image_ndarray = None
            self._image_daskarray = None

        if disk:
            cache_path = cache_path or (
                _ImagingDatasetCache.generate_cache_path(self.directory,
                                                         self.id))
            log.debug('Attempting to delete cache file at: "{}"', cache_path)
            try:
                os.remove(cache_path)
            except FileNotFoundError as e:
                log.debug('Cache file not found.', exc_info=e)
                log.info('Cache file not found.')
            else:
                log.info('Cache file deleted.')

    def imsave(self,
               path=None,
               *,
               name_prefix='',
               extension=None,
               _image_data=None,
               **kwargs):

        if path is None:
            name_str = self.id
            if self.path is not None:
                path = os.path.join(self.directory, name_str)
            else:
                config = get_global_config()
                name_str = self.id
                path = os.path.join(config.output_directory, name_str)

        if extension is not None:
            path += extension

        _dir, _name = os.path.split(path)
        path = os.path.join(_dir, (name_prefix + _name))

        log.debug('Getting imaging data for saving.')
        if _image_data is None:
            image_data = self.get_image_data()
        else:
            image_data = _image_data

        log.info('Saving dataset imaging data to an image file.')
        log.debug('Saving dataset\'s image to "{}".', path)
        skimage.io.imsave(path, image_data, **kwargs)

    def to_tiff(self, path=None, use_imagej_dim_order=True, **kwargs):
        tiff_kwargs = kwargs
        tiff_kwargs['plugin'] = 'tifffile'
        tiff_kwargs.setdefault('imagej', True)
        tiff_kwargs.setdefault('check_contrast', False)

        imsave_kwargs = {'path': path,
                         'extension': TIFF_EXTENSION}

        # FIXME - need to convert the image data to one of the following
        #   data types: 'B', 'H', 'h', 'f' (look up what these mean) for ImageJ
        #   compatibility. See `tifffile.TiffWriter.write(...)` source code
        #   for more information

        if use_imagej_dim_order:
            log.info('Transposing dataset dimensions for creating a tiff '
                     'readable in ImageJ.')

            image_j_dim_order = (
                Dim.POSITION,
                Dim.TIME,
                Dim.Z,
                Dim.CHANNEL,
                Dim.Y,
                Dim.X)

            image_data = self.get_image_data(in_standard_dims=True)

            new_axes_order = [default_dim_order_dict[d]
                              for d in image_j_dim_order]
            image_data = np.transpose(image_data, axes=new_axes_order)

            # FIXME - make into a `squeeze_leading` function
            squeeze_list = []
            _iter = enumerate(zip(default_dimension_order, image_data.shape))
            for i, (d, s) in _iter:
                if d in default_page_dimension_order:
                    break
                if s == 1:
                    squeeze_list.append(i)
                else:
                    break

            if image_data.shape[0] > 1 and image_data.shape[1] > 1:
                msg = 'Cannot handle multi-position + multi-time data.'
                raise NotImplementedError(msg)
            elif image_data.shape[0] > 1 and image_data.shape[1] == 1:
                log.warning('Play axis in ImageJ for this dataset will '
                            'represent position, not time.')
                squeeze_list.append(1)

            squeeze_list = tuple(squeeze_list)
            if squeeze_list:
                image_data = np.squeeze(image_data, axis=squeeze_list)
            self.imsave(_image_data=image_data,
                        **imsave_kwargs,
                        **tiff_kwargs)
        else:
            self.imsave(**imsave_kwargs, **tiff_kwargs)

    def get_image_data(self, lazy=True, in_standard_dims=False):
        if self._image_ndarray is not None:
            out_data = self._image_ndarray
        elif self._image_daskarray is not None:
            if lazy:
                out_data = self._image_daskarray
            else:
                out_data = self._image_daskarray.compute()
        elif self._get_image_data_func is not None:
            log.info('Running function for retrieving/computing image data.')
            out_data = self._get_image_data_func()
            if isinstance(out_data, np.ndarray):
                self._image_ndarray = out_data
                self._did_initialize_image_data = True
            elif isinstance(out_data, dask.array.Array):
                self._image_daskarray = out_data
                self._did_initialize_image_data = True
        else:
            msg = 'No image data has been set for this ImagingDataset.'
            RuntimeError(msg)

        if in_standard_dims:
            standard_shape = []
            for d in default_dimension_order:
                try:
                    dim_size = self.shape_dict[d]
                except KeyError:
                    dim_size = 1
                standard_shape.append(dim_size)
            out_data = out_data.reshape(standard_shape)

        return out_data

    def add_to_napari(self,
                      napari_viewer,
                      translate=None,
                      with_world_extents=None,
                      **kwargs):

        log.debug('{} | translate = {}', repr(self), str(translate))

        if isinstance(translate, dict):
            translation_vector = []
            for d in default_dimension_order:
                if d is Dim.CHANNEL:
                    translation_vector.append(None)
                    # this will raise an error if we don't remove the entry
                    # later
                    continue
                try:
                    translation_vector.append(translate[d])
                except KeyError:
                    translation_vector.append(0)
            translate = translation_vector

        log.debug('{} | translate = {}', repr(self), str(translate))

        image_data = self.get_image_data(in_standard_dims=True)

        scale = self.standard_scale

        if with_world_extents is not None:
            log.info('Extending image into world extents for viewing in '
                     'napari.')
            image_data, scale, translate = self._scale_singleton_dims_to_extent(
                image_data,
                dimensions=default_dimension_order,
                extents=with_world_extents,
                scale=self.standard_scale,
                translate=translate)

        log.debug('{} | translate = {}', repr(self), str(translate))

        non_channel_scale = []
        for s, d in zip(scale, default_dimension_order):
            if d is Dim.CHANNEL:
                continue
            non_channel_scale.append(s)
        non_channel_scale = tuple(non_channel_scale)

        if translate is not None:
            non_channel_translate = []
            for t, d in zip(translate, default_dimension_order):
                if d is Dim.CHANNEL:
                    continue
                non_channel_translate.append(t)
        else:
            non_channel_translate = None

        log.debug('{} | non_channel_translate = {}', repr(self),
                  str(non_channel_translate))

        contrast_range = [np.min([image_data.min(), 0]),
                          image_data.max()]

        napari_viewer.add_image(
            name=self.name,
            data=image_data,
            scale=non_channel_scale,
            channel_axis=default_dim_order_dict[Dim.CHANNEL],
            translate=non_channel_translate,
            contrast_limits=contrast_range,
            **kwargs)

    def cache_to_disk(self, *, lazy=True):
        log.info('Caching imaging dataset to disk.')
        cache_object = _ImagingDatasetCache(self, lazy=lazy)

        cache_path = cache_object.generate_cache_path(self.directory, self.id)

        cache_dir = os.path.split(cache_path)[0]
        csutils.touchdir(cache_dir, recur=False)

        log.debug('Saving _ImagingDatasetCache to disk at "{}".',
                  cache_path)
        if os.path.exists(cache_path):
            log.debug('Overwriting existing cache data.')
        with open(cache_path, 'wb') as _file:
            pickle.dump(cache_object, _file)

    def _init_from_cache(self,
                         *,
                         cache_path=None):
        log.info('Attempting to load imaging dataset from cache.')
        if cache_path is None:
            cache_path = _ImagingDatasetCache.generate_cache_path(
                self.directory, self.id)

        log.debug('Loading _ImagingDatasetCache from disk at "{}".',
                  cache_path)
        try:
            with open(cache_path, 'rb') as _file:
                cache_object: _ImagingDatasetCache = pickle.load(_file)
        except FileNotFoundError as e:
            log.info('Cache file not found.')
            log.debug('Cache file not found error info:', exc_info=e)
            return
        except Exception as e:
            log.info('Cache file found, but could not be loaded.')
            log.debug('Error caught when loading cache file:', exc_info=e)
            return

        if isinstance(cache_object.image_data, np.ndarray):
            self._image_ndarray = cache_object.image_data
            self._did_initialize_image_data = True
        else:
            msg = 'Got an expected type from cache_object.image_data.'
            RuntimeError(msg)

        if not self._did_initialize_image_data:
            log.warning('Cache file did not contain usable image data; '
                        'ignoring cache.')
            return

        mismatch_msg = ('Cache object and object under initialization '
                        'have different values for property "{}". '
                        '(cache object: {}, init: {}. Using the {} value.')

        if self.name is None:
            self.name = cache_object.name
        elif not (self.name == cache_object.name):
            log.warning(mismatch_msg, 'name', cache_object.name, self.name,
                        'init')

        if self.dimensions is None:
            self.dimensions = cache_object.dimensions
        elif not (self.dimensions == cache_object.dimensions):
            log.warning(mismatch_msg, 'dimensions', cache_object.dimensions,
                        self.dimensions, 'cache')
            self.dimensions = cache_object.dimensions

        if self.coordinates is None:
            self.coordinates = cache_object.coordinates
        elif not (self.coordinates == cache_object.coordinates):
            log.warning(mismatch_msg, COORDINATES_KEY, cache_object.coordinates,
                        self.coordinates, 'cache')
            self.coordinates = cache_object.coordinates

        if self.position is None:
            self.position = cache_object.position
        elif not (self.position == cache_object.position):
            log.warning(mismatch_msg, 'position', cache_object.position,
                        self.position, 'cache')
            self.position = cache_object.position

    # getters and setters
    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def axis_dict(self) -> dict[Dimension, int]:
        # if self.dimensions is None or the dimensions dict is empty
        if not self.dimensions:
            return dict()

        idx: Iterable[int]
        dim: Iterable[Dimension]
        idx, dim = zip(*enumerate(self.dimensions))
        return dict(zip(dim, idx))
            
    @property
    def directory(self):
        if os.path.isdir(self.path):
            return self.path
        else:
            return os.path.split(self.path)[0]

    @property
    def id(self):
        if self._id is None:
            if self.source_data is None:
                return self.name or repr(self)
            else:
                return (f'{self.source_data.__class__.__name__}-'
                        f'{self.name or repr(self)}')
        else:
            return self._id

    @property
    def shape(self):
        # FIXME - let's see if there's a more effecient way to do this
        return self.get_image_data().shape

    @property
    def shape_dict(self):
        return dict(zip(self.dimensions, self.shape))

    @property
    def scale(self):
        scale = []
        for d in self.dimensions:
            c = self.coordinates[d]
            scale.append(c.median_pitch)
        return tuple(scale)

    @property
    def scale_dict(self):
        return dict(zip(self.dimensions, self.scale))

    @property
    def standard_scale(self):
        std_scale = []
        for d in default_dimension_order:
            try:
                std_scale.append(self.scale_dict[d])
            except KeyError:
                std_scale.append(SCALE_PLACEHOLDER)
        return tuple(std_scale)

    @property
    def extents(self):
        extents = dict()
        for d in self.dimensions:
            if not d.is_numeric:
                extents[d] = None
                continue

            if self.coordinates is None:
                continue

            try:
                coord = self.coordinates[d]
            except KeyError:
                continue

            try:
                pos = self.position[d]
            except KeyError:
                pos = 0

            half_pitch = (coord.median_pitch / 2)
            low_bound = pos + min(coord.vector) - half_pitch
            high_bound = pos + max(coord.vector) + half_pitch
            extents[d] = [low_bound, high_bound]
        return extents

    # class methods
    @classmethod
    def from_dataset_info(cls,
                          dataset_info: ZMIADatasetInfo,
                          loader_kwargs: dict[str, Any] = None,
                          **kwargs) -> ImagingDataset:
        loader = choose_dataset_loader(dataset_info=dataset_info)
        return cls.from_dataset(
            loader(
                dataset_info=dataset_info,
                **(loader_kwargs or dict())),
            path=dataset_info.config.output_directory,
            **kwargs)

    @classmethod
    def from_dataset(cls,
                     dataset: (pvi.PVDataset | AVDataset | NISDataset
                               | TIFFDataset),
                     **kwargs) -> ImagingDataset:
        if isinstance(dataset, pvi.PVDataset):
            return cls.from_pvdataset(dataset, **kwargs)
        elif isinstance(dataset, AVDataset):
            return cls.from_video_dataset(dataset, **kwargs)
        elif isinstance(dataset, NISDataset):
            return cls.from_nisdataset(dataset, **kwargs)
        elif isinstance(dataset, TIFFDataset):
            return cls.from_tiff_dataset(dataset, **kwargs)
        else:
            msg = (f'Unexpected dataset type passed when attempting '
                   f'to create an imaging dataset, got instance of '
                   f'{dataset.__class__.__name__}.')
            raise RuntimeError(msg)

    @classmethod
    def from_tiff_dataset(cls,
                          tiff_ds,
                          **kwargs
                          ) -> ImagingDataset:

        import tiff_utils

        init_kwargs: dict[str, Any] = kwargs

        try:
            dimensions: Optional[tuple[Dim, ...]] = init_kwargs['dimensions']
            # FIXME - might need to handle this differently for when `None`
            #   is explicitly passed in `kwargs` for dimensions kw argument
            #   same for `coordinates`
        except KeyError:
            dimensions = None

        if dimensions is not None:
            pass  # already set by kwargs
        elif tiff_ds.dimensions is not None:
            dimensions = tiff_ds.dimensions
        else:
            if tiff_ds.ndim == 2:
                dimensions = (Dim.Y, Dim.X)
            elif tiff_ds.ndim == 3:
                dimensions = (Dim.Z, Dim.Y, Dim.X)
            else:
                msg = (f'Cannot make assumption of the tiff datasets dimension '
                       f'order; encountered tiff file with {tiff_ds.ndim = } .')
                log.warning(msg)

            if dimensions is not None:
                log.warning('Assuming dimension order for tiff with {:d}'
                            ' dimensions is {}. ', tiff_ds.ndim, dimensions)

        try:
            coordinates: Optional[dict[Dim, CoordinateVector]] = (
                init_kwargs[COORDINATES_KEY])
        except KeyError:
            coordinates = None

        if coordinates is not None:
            pass  # already set by kwargs
        elif dimensions is None:
            pass  # nothing to base coordinate dict on
        else:
            for d, d_size in zip(dimensions, tiff_ds.shape):
                pitch_um: Optional[float] = None
                if d in (Dim.X, Dim.Y):
                    pitch_um = tiff_ds.xy_pitch_um
                elif d is Dim.Z:
                    pitch_um = tiff_ds.z_pitch_um
                else:
                    log.warning('Cannot assign coordinates for dimension {}'
                                ' when converting TiffDataset to '
                                'ImagingDataset', d)

                if pitch_um is None:
                    continue

                if coordinates is None:
                    coordinates = dict()

                coord_ndarray = np.arange(d_size) * pitch_um
                coordinates[d] = CoordinateVector(
                    vector=coord_ndarray,
                    unit=tiff_utils.LENGTH_UNITS)

        image_data_func: Callable[[], np.ndarray] = (
            lambda: tiff_ds.image_ndarray)

        init_kwargs.setdefault('path', tiff_ds.path)
        init_kwargs.setdefault('name', tiff_ds.name)
        init_kwargs.setdefault('cache_after_init', True)

        obj = cls(
            dimensions=dimensions,
            coordinates=coordinates,
            get_image_data_func=image_data_func,
            source_data=tiff_ds,
            **init_kwargs)

        return obj

    @classmethod
    def from_nisdataset(cls, nd2ds: NISDataset, **kwargs):
        dimensions = ([nd2_str_to_dim[ax] for ax in nd2ds.non_page_axes]
                      + [Dim.Y, Dim.X])

        coordinates = dict()
        coordinates[Dim.X] = CoordinateVector(
            np.arange(nd2ds.page_shape[-1]) * nd2ds.pixel_pitch_um,
            pvi.POSITION_UNITS)
        coordinates[Dim.Y] = CoordinateVector(
            np.arange(nd2ds.page_shape[0]) * nd2ds.pixel_pitch_um,
            pvi.POSITION_UNITS)

        for d in dimensions[:-2]:
            if d is Dim.CHANNEL:
                coordinates[d] = CoordinateVector(
                    nd2ds.channel_name_list,
                    None)
            elif d is Dim.TIME:
                raise NotImplementedError('Cannot parse time for nisdataset '
                                          'conversion.')
            elif d is Dim.POSITION:
                coordinates[d] = CoordinateVector(
                    np.array(list(range(nd2ds.sizes[Dim.POSITION.nd2_str]))),
                    None)
            elif d is Dim.Z:
                coordinates[d] = CoordinateVector(
                    np.array(nd2ds.z_locs),
                    pvi.POSITION_UNITS)

        init_kwargs = kwargs
        init_kwargs['get_image_data_func'] = lambda: nd2ds.image_ndarray
        init_kwargs['dimensions'] = dimensions
        init_kwargs[COORDINATES_KEY] = coordinates
        init_kwargs['source_data'] = nd2ds
        init_kwargs.setdefault('cache_after_init', True)
        init_kwargs.setdefault('name', nd2ds.name)
        init_kwargs.setdefault('path', nd2ds.path)
        return cls(**init_kwargs)

    @classmethod
    def from_video_dataset(cls,
                           vid_ds: AVDataset,
                           **kwargs):
        dimensions = [Dim.TIME, Dim.Y, Dim.X]
        coordinates = dict()
        coordinates[Dim.TIME] = CoordinateVector(
            np.arange(vid_ds.num_frames) * vid_ds.frame_period_second,
            pvi.FLOAT_TIME_UNITS)
        if vid_ds.pixel_pitch_um is not None:
            coordinates[Dim.Y] = CoordinateVector(
                np.arange(vid_ds.shape[1]) * vid_ds.pixel_pitch_um,
                pvi.POSITION_UNITS)
            coordinates[Dim.X] = CoordinateVector(
                np.arange(vid_ds.shape[2]) * vid_ds.pixel_pitch_um,
                pvi.POSITION_UNITS)

        init_kwargs = kwargs
        init_kwargs['get_image_data_func'] = lambda: vid_ds.image_ndarray
        init_kwargs['dimensions'] = dimensions
        init_kwargs[COORDINATES_KEY] = coordinates
        init_kwargs['source_data'] = vid_ds
        init_kwargs.setdefault('cache_after_init', True)
        init_kwargs.setdefault('name', vid_ds.name)
        init_kwargs.setdefault('path', vid_ds.path)
        return cls(**init_kwargs)

    # @csutils.timed(logger=log)
    @classmethod
    def from_pvdataset(cls, pvdataset: pvi.PVDataset, **kwargs):
        """
        initialize an imaging dataset from a pv dataset
        """

        tt = time.time()
        if not isinstance(pvdataset, pvi.PVDataset):
            msg = (f'pvdataset must be type {pvi.PVDataset.__name__}, got '
                   f'{pvdataset.__class__.__name__} instead.')
            raise TypeError(msg)

        if not pvdataset:
            msg = (f'{pvdataset} is an empty dataset and cannot be used to '
                   f'initialize an {cls.__name__} object.')
            raise ValueError(msg)

        category = pvdataset.category
        log.info('Categorized PVDataset as {}', category)

        if category is pvi.PVDatasetCategory.UNKNOWN:
            msg = (f'{pvdataset} cannot be categorized into into a known'
                   f'conversion schema; {cls.__name__} instantiation failed.')
            raise ValueError(msg)

        log.info('Initializing ImagingDataset(s) from prairie view dataset '
                 '{}', pvdataset)

        init_kwargs = kwargs
        init_kwargs['source_data'] = pvdataset
        init_kwargs.setdefault('path', pvdataset.directory)
        init_kwargs.setdefault('name', pvdataset.name)

        im_dataset = cls(check_for_image_data=False, **init_kwargs)
        if im_dataset._did_initialize_image_data:
            # successfully loaded data from cache
            return im_dataset

        init_kwargs.setdefault('cache_after_init', True)

        pvdataset.load_all_image_data()

        log.info('Using dataset category to convert to imaging dataset.')
        if category is pvi.PVDatasetCategory.SINGLE_IMAGE:
            single_2d_image = pvdataset.all_2d_images[0]
            log.info('Generating image_ndarray from raw dataset.')
            image_ndarray = single_2d_image.image_ndarray
            return cls(image_ndarray=image_ndarray,
                       dimensions=default_page_dimension_order,
                       **init_kwargs)

        if category is pvi.PVDatasetCategory.MULTI_CHANNEL_SINGLE_IMAGE:
            log.info('Generating image_ndarray from raw dataset.')
            image_ndarray_list = [pv_2d_im.image_ndarray
                                  for pv_2d_im in pvdataset.all_2d_images]
            image_ndarray = np.concatenate(image_ndarray_list,
                                           axis=2)
            dims = (Dim.CHANNEL, ) + default_page_dimension_order
            return cls(image_ndarray=image_ndarray,
                       dimensions=dims,
                       **init_kwargs)

        md_reader = pvi.PVMetadataInterpreter(pvdataset.metadata)

        if pvi.PVDatasetCategory.TZ_SERIES_IMAGE in category:
            if pvi.PVDatasetCategory.MULTI_CHANNEL in category:
                is_multi_channel = True
                msg = 'Cannot handle multi-channel TZ images.'
                raise NotImplementedError(msg)


            log.info('Processing multi-dimensional data and extracting'
                     ' coordinate information.')
            sizes = dict()
            sizes[Dim.X] = md_reader.x_size
            sizes[Dim.Y] = md_reader.y_size
            sizes[Dim.Z] = md_reader.z_size
            sizes[Dim.TIME] = md_reader.num_sequences

            # TODO - channel will need to be incorporated here and throughout
            unraveled_non_page_size = sizes[Dim.Z] * sizes[Dim.TIME]
            unraveled_shape = (unraveled_non_page_size,
                               sizes[Dim.Y],
                               sizes[Dim.X])

            coordinates = dict()
            coordinates[Dim.X] = CoordinateVector(
                md_reader.x_coord_array, pvi.POSITION_UNITS)
            coordinates[Dim.Y] = CoordinateVector(
                md_reader.y_coord_array, pvi.POSITION_UNITS)
            coordinates[Dim.Z] = CoordinateVector(
                md_reader.z_coord_array, pvi.POSITION_UNITS)
            coordinates[Dim.TIME] = CoordinateVector(
                md_reader.starting_relative_time_coord_array,
                pvi.FLOAT_TIME_UNITS)
            init_kwargs[COORDINATES_KEY] = coordinates

            position = dict()
            position[Dim.X] = md_reader.x_left_position
            position[Dim.Y] = md_reader.y_top_position
            position[Dim.Z] = md_reader.z_top_position
            position[Dim.TIME] = md_reader.time_first_position
            init_kwargs['position'] = position

            parse_dim_order = (Dim.TIME, Dim.Z)
            full_dim_order = parse_dim_order + default_page_dimension_order
            init_kwargs['dimensions'] = full_dim_order

            full_dataset_shape = tuple(sizes[d] for d in full_dim_order)

            z_locs = md_reader.get_unique_net_z_locations(
                sorting=pvi.MetadataSorting.VALUE)

            all_2d_images = pvdataset.all_2d_images
            im_ids_by_frame_id = md_reader.get_im_ids_by_sequence()
            im_ids_by_z = md_reader.get_im_ids_by_net_z_location()

            image_filter_args = [pvi.pvmk.TWO_D_IMAGE_ID, None]

            # FIXME - the double filtering is ineffecient

            log.info('Generating image_ndarray from raw dataset.')
            unraveled_image_list = []
            for frame_id in md_reader.sequence_index:
                log.debug('Gathering image data from frame_id = {}',
                          frame_id)
                
                image_ids = im_ids_by_frame_id[frame_id]
                image_filter_args[1] = image_ids
                seq_2d_images = \
                    all_2d_images.filter_within_metadata_value_list(
                        *image_filter_args)

                for z_loc in z_locs:
                    image_ids = im_ids_by_z[z_loc]
                    image_filter_args[1] = image_ids
                    single_2d_image = \
                        seq_2d_images.filter_within_metadata_value_list(
                            *image_filter_args)

                    assert len(single_2d_image) == 1, (
                        f'Expected to receive a scalar PVDataList here; '
                        f'got PVDataList of len {len(single_2d_image):d} '
                        f'instead.')

                    unraveled_image_list.append(
                        single_2d_image[0].image_ndarray)

            ref_im = unraveled_image_list[0]
            unraveled_ndarray = np.zeros_like(ref_im,
                                              shape=unraveled_shape)
            np.stack(unraveled_image_list,
                     axis=0,
                     out=unraveled_ndarray)
            if unraveled_shape == full_dataset_shape:
                image_ndarray = unraveled_ndarray
            else:
                image_ndarray = unraveled_ndarray.reshape(full_dataset_shape)

            dataset = cls(image_ndarray=image_ndarray, **init_kwargs)

            log.info('Conversion into imaging dataset and structuring of '
                     'ndarray complete.')

            return dataset

        if pvi.PVDatasetCategory.T_SERIES in category:
            is_multi_channel = pvi.PVDatasetCategory.MULTI_CHANNEL in category

            # FIXME - there's a better way to do this (getting example value)
            channel_indexes = md_reader.get_channel_index_by_frame()[0]

            log.info('Processing multi-dimensional data and extracting'
                     ' coordinate information.')

            sizes = dict()
            sizes[Dim.X] = md_reader.x_size
            sizes[Dim.Y] = md_reader.y_size
            sizes[Dim.TIME] = md_reader.num_frames

            if is_multi_channel:
                sizes[Dim.CHANNEL] = len(channel_indexes)
                unraveled_non_page_size = sizes[Dim.CHANNEL] * sizes[Dim.TIME]
            else:
                unraveled_non_page_size = sizes[Dim.TIME]
            unraveled_shape = (
                unraveled_non_page_size,
                sizes[Dim.Y],
                sizes[Dim.X])

            coordinates = dict()
            coordinates[Dim.X] = CoordinateVector(
                md_reader.x_coord_array, pvi.POSITION_UNITS)
            coordinates[Dim.Y] = CoordinateVector(
                md_reader.y_coord_array, pvi.POSITION_UNITS)
            coordinates[Dim.TIME] = CoordinateVector(
                md_reader.starting_relative_time_coord_array,
                pvi.FLOAT_TIME_UNITS)
            if is_multi_channel:
                # FIXME - add channel descriptions
                channel_index_dict = md_reader \
                    .get_channel_name_by_channel_index()
                raise NotImplementedError('Cannot get channel indexes '
                                          'descriptions for coordinate spec.')
            init_kwargs[COORDINATES_KEY] = coordinates

            position = dict()
            position[Dim.X] = md_reader.x_left_position
            position[Dim.Y] = md_reader.y_top_position
            position[Dim.TIME] = md_reader.time_first_position
            init_kwargs['position'] = position

            parse_dim_order = (Dim.TIME, )
            full_dim_order = parse_dim_order + default_page_dimension_order
            init_kwargs['dimensions'] = full_dim_order

            full_dataset_shape = tuple(sizes[d] for d in full_dim_order)

            all_2d_images = pvdataset.all_2d_images
            im_ids_by_frame_id = md_reader.get_im_ids_by_frame_id()
            frame_ids = md_reader.frame_id[0]  # we know there's only one seq.

            image_filter_args = [pvi.pvmk.TWO_D_IMAGE_ID, None]

            # FIXME - the double filtering is ineffecient

            log.info('Generating image_ndarray from raw dataset.')
            unraveled_image_list = []
            for frame_id in frame_ids:
                image_ids = im_ids_by_frame_id[frame_id]
                image_filter_args[1] = image_ids
                single_2d_image = \
                    all_2d_images.filter_within_metadata_value_list(
                        *image_filter_args)

                assert len(single_2d_image) == 1, (
                    f'Expected to receive a scalar PVDataList here; '
                    f'got PVDataList of len {len(single_2d_image):d} instead.')

                unraveled_image_list.append(single_2d_image[0].image_ndarray)

            ref_im = unraveled_image_list[0]
            unraveled_ndarray = np.zeros_like(ref_im,
                                              shape=unraveled_shape)
            np.stack(unraveled_image_list,
                     axis=0,
                     out=unraveled_ndarray)
            if unraveled_shape == full_dataset_shape:
                image_ndarray = unraveled_ndarray
            else:
                image_ndarray = unraveled_ndarray.reshape(full_dataset_shape)

            dataset = cls(image_ndarray=image_ndarray, **init_kwargs)

            log.info('Conversion into imaging dataset and structuring of '
                     'ndarray complete.')

            return dataset

        if pvi.PVDatasetCategory.Z_STACK_IMAGE in category:
            if pvi.PVDatasetCategory.MULTI_TRACK in category:
                has_tracks = True
                # FIXME - there's a better way to do this
                track_index_dict = \
                    md_reader.get_track_index_by_net_z_location()
                track_indexes = list(track_index_dict.values())[0]
            else:
                track_indexes = [None]
                has_tracks = False

            log.info('Processing multi-dimensional data and extracting'
                     ' coordinate information.')

            z_locs = md_reader.get_unique_net_z_locations(
                sorting=pvi.MetadataSorting.VALUE)

            is_multi_channel = pvi.PVDatasetCategory.MULTI_CHANNEL in category

            # FIXME - there's a better way to do this (getting example value)
            channel_indexes = md_reader.get_channel_index_by_frame()[0]

            sizes = dict()
            sizes[Dim.X] = md_reader.x_size
            sizes[Dim.Y] = md_reader.y_size
            sizes[Dim.Z] = len(z_locs)

            if is_multi_channel:
                sizes[Dim.CHANNEL] = len(channel_indexes)
                unraveled_non_page_size = sizes[Dim.CHANNEL] * sizes[Dim.Z]
            else:
                unraveled_non_page_size = sizes[Dim.Z]

            unraveled_shape = (
                unraveled_non_page_size,
                sizes[Dim.Y],
                sizes[Dim.X])

            coordinates = dict()
            coordinates[Dim.X] = CoordinateVector(
                md_reader.x_coord_array, pvi.POSITION_UNITS)
            coordinates[Dim.Y] = CoordinateVector(
                md_reader.y_coord_array, pvi.POSITION_UNITS)
            coordinates[Dim.Z] = CoordinateVector(
                md_reader.z_coord_array, pvi.POSITION_UNITS)
            if is_multi_channel:
                # FIXME - add channel descriptions
                channel_index_dict = md_reader\
                    .get_channel_name_by_channel_index()
                raise NotImplementedError('Cannot get channel indexes '
                                          'descriptions for coordinate spec.')
            init_kwargs[COORDINATES_KEY] = coordinates

            position = dict()
            position[Dim.X] = md_reader.x_left_position
            position[Dim.Y] = md_reader.y_top_position
            position[Dim.Z] = md_reader.z_top_position
            init_kwargs['position'] = position

            all_2d_images = pvdataset.all_2d_images
            im_id_dict = md_reader.get_im_ids_by_net_z_location()

            image_filter_args = [pvi.pvmk.TWO_D_IMAGE_ID, None]
            channel_filter_args = [pvi.pvmk.CHANNEL_INDEX, None]
            track_filter_args = [pvi.pvmk.TRACK_INDEX, None]

            parse_dim_order = (Dim.Z, )
            if is_multi_channel:
                parse_dim_order += (Dim.CHANNEL, )
            full_dim_order = parse_dim_order + default_page_dimension_order
            init_kwargs['dimensions'] = full_dim_order
            full_dataset_shape = tuple(sizes[d] for d in full_dim_order)

            log.info('Generating image_ndarray from raw dataset.')
            datasets = dict()
            for track_i in track_indexes:
                if has_tracks:
                    track_filter_args[1] = track_i
                    track_2d_images = all_2d_images.filter_with_metadata_value(
                        *track_filter_args)
                else:
                    track_2d_images = all_2d_images

                unraveled_image_list = []
                for z_loc in z_locs:
                    image_ids = im_id_dict[z_loc]
                    image_filter_args[1] = image_ids
                    z_2d_images = \
                        track_2d_images.filter_within_metadata_value_list(
                            *image_filter_args)

                    for channel_index in channel_indexes:
                        channel_filter_args[1] = channel_index
                        single_2d_image = \
                            z_2d_images.filter_with_metadata_value(
                                *channel_filter_args)

                        assert len(single_2d_image) == 1, (
                            f'Expected to receive a scalar PVDataList here; '
                            f'got list of len {len(single_2d_image):d} '
                            f'instead.')

                        unraveled_image_list.append(
                            single_2d_image[0].image_ndarray)

                ref_im = unraveled_image_list[0]
                unraveled_ndarray = np.zeros_like(ref_im,
                                                  shape=unraveled_shape)
                np.stack(unraveled_image_list,
                         axis=0,
                         out=unraveled_ndarray)
                if unraveled_shape == full_dataset_shape:
                    image_ndarray = unraveled_ndarray
                else:
                    image_ndarray = unraveled_ndarray\
                        .reshape(full_dataset_shape)

                if has_tracks:
                    track_init_kwargs = copy.deepcopy(init_kwargs)
                    track_init_kwargs['name'] = (
                        f'Track{track_i:d}-{init_kwargs["name"]}')
                else:
                    track_init_kwargs = init_kwargs

                datasets[track_i] = cls(image_ndarray=image_ndarray,
                                        **track_init_kwargs)

            log.info('Conversion into imaging dataset and structuring of '
                     'ndarray(s) complete.')

            if has_tracks:
                log.warning('Returning multi-track PVDataset as a dict of '
                            'ImagingDataset\'s indexed by their track index.')
                return ImagingDatasetDict(
                    datasets,
                    key_type=DatasetListKeyType.PRAIRIE_VIEW_TRACK)
            else:
                return datasets[None]

    # static methods
    @staticmethod
    def apply_standard_napari_config(napari_viewer):
        non_channel_dim_order = []
        for d in default_dimension_order:
            if d is Dim.CHANNEL:
                continue
            non_channel_dim_order.append(d)

        napari_viewer.dims.axis_labels = non_channel_dim_order
        napari_viewer.dims.order = tuple(range(len(non_channel_dim_order)))
        napari_viewer.scale_bar.visible = True
        napari_viewer.reset_view()

    @staticmethod
    def _scale_singleton_dims_to_extent(image_data,
                                        *,
                                        scale,
                                        dimensions,
                                        extents,
                                        translate=None):
        scale = list(scale)
        if translate is None:
            translate = [0 for _ in dimensions]
        else:
            translate = list(translate)

        for axis_idx, (d, shape) \
                in enumerate(zip(dimensions, image_data.shape)):

            if (d not in extents) or not (shape == 1):
                continue

            image_data = np.concatenate((image_data, image_data),
                                        axis=axis_idx)

            scale[axis_idx] = (extents[d][1] - extents[d][0]) / 2
            translate[axis_idx] = (extents[d][0] + translate[axis_idx]
                                   + (scale[axis_idx] / 2))

        return image_data, tuple(scale), translate


class _ImagingDatasetCache(object):
    """
    helper class for dumping/restoring an ImagingDataset object to/from the disk
    """

    CACHE_FILENAME_FMT = '.{:s}_imd-cache' + PICKLE_EXTENSION

    def __init__(self, dataset: ImagingDataset, *, lazy=False):
        self.name = dataset.name
        self.image_data = dataset.get_image_data(lazy=lazy)
        self.dimensions = dataset.dimensions
        self.coordinates = dataset.coordinates
        self.position = dataset.position

    @classmethod
    def generate_cache_path(cls, directory, cache_id):
        return os.path.join(directory,
                            CACHE_DIR_NAME,
                            cls.CACHE_FILENAME_FMT.format(cache_id))


class DatasetListKeyType(enum.Enum):
    PRAIRIE_VIEW_TRACK = enum.auto()


class ImagingDatasetDict(UserDict):
    def __init__(self, *args, key_type, **kwargs):
        super().__init__(*args, **kwargs)
        self.key_type = key_type






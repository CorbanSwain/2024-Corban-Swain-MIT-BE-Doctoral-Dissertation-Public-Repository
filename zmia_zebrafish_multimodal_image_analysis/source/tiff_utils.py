#!python3
# tiff_utils.py

# imports
from __future__ import annotations
from typing import Union, Any, Optional
import os
import numpy as np
import tifffile
import contextlib

# - local imports
import c_swain_python_utils as csutils
import utilities
from utilities import *


# CONSTANTS
class TiffKeys:
    BITS_PER_SAMPLE = 'BitsPerSample'


zdik = ZmiaDatasetInfoKeys

LENGTH_UNITS = 'micrometer'

# logging
log = csutils.get_logger(__name__)


class TIFFDataset(object):

    def __init__(self,
                 path: str | bytes | os.PathLike[Any],
                 *,
                 name: Optional[str] = None,
                 image_data: Optional[np.ndarray] = None,
                 lazy: bool = False
                 ):

        self.path = path
        self.name = name
        self.xy_pitch_um: Optional[float] = None
        self.z_pitch_um: Optional[float] = None
        self.bits_per_sample: Optional[int] = None
        self.dimensions: Optional[tuple[Dim, ...]] = None
        self._image_ndarray = image_data
        self._metadata: tifffile.TiffTags

        self._init_from_tifffile(lazy=lazy)

    # properties
    @property
    def ndim(self) -> int:
        return self.image_ndarray.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.image_ndarray.shape

    @property
    def image_ndarray(self) -> np.ndarray:
        """
        ndarray of the image data
        """
        self._load_image_data()
        return self._image_ndarray

    @property
    def metadata(self) -> tifffile.TiffTags:
        """
        metadata from tiff file as tifffile.TiffTags
        """
        return self._metadata

    # protected methods
    @contextlib.contextmanager
    def _tifffile(self) -> tifffile.TiffFile:
        """
        wrapper for tifffile.TiffFile context using the objects path
        """
        with tifffile.TiffFile(self.path) as tif:
            yield tif

    def _init_from_tifffile(self,
                            *,
                            lazy: bool):
        """
        initializes the object attributes from a tiff file on disk
        """
        if not lazy:
            self._load_image_data()

        with self._tifffile() as tif:
            self._metadata = tif.pages[0].tags
            self.bits_per_sample = (
                    self.metadata[TiffKeys.BITS_PER_SAMPLE].value)

    def _load_image_data(self,
                         force: bool = False) -> None:
        """
        loads image data from the tifffile into an ndarray attribute of the
        object. use force=True to reload the data from the file.
        """
        if (self._image_ndarray is not None) and (not force):
            log.debug('Image data already loaded.')
            return

        log.info('Loading all image data from tiff file at "{:s}".',
                 self.path)
        with self._tifffile() as tif:
            self._image_ndarray = tif.asarray()

    # class methods
    @classmethod
    def from_dataset_info(cls,
                          *,
                          dataset_info: ZMIADatasetInfo,
                          **kwargs
                          ) -> TIFFDataset:
        obj = cls(dataset_info.full_path,
                  name=dataset_info.name)
        obj.xy_pitch_um = dataset_info[zdik.XY_PITCH_UM]
        obj.z_pitch_um = dataset_info[zdik.Z_PITCH_UM]
        return obj




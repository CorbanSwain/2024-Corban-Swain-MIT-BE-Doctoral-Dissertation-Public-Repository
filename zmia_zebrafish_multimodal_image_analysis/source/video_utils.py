#!python3
# video_utils.py
from __future__ import annotations

import collections
# general imports
import contextlib
import fractions
import os
import shutil
import subprocess
import time
from typing import Generator, Tuple, Type, Dict, Union, Optional, List, \
    NamedTuple
from types import ModuleType
import numpy as np
import datetime as dt

# local modules
import c_swain_python_utils as csutils
import utilities
from utilities import *

# attempt import of av backend dependencies
try:
    import pims
except ImportError:
    pims = None
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import avi_r
except ImportError:
    avi_r = None
try:
    import av
except ImportError:
    av = None


log = csutils.get_logger(__name__)


class Frame(object):
    def __init__(self, frame_data, *, is_error, exc_info=None, index=None):
        self.frame_data = frame_data
        self.is_error = is_error
        self.exc_info = exc_info
        self.index = index

    def __bool__(self):
        return self.is_error

    def __repr__(self):
        return (f'{self.__class__.__name__}<'
                f'is_error={self.is_error}, '
                f'exc_info={self.exc_info}, '
                f'index={self.index}>')

    @property
    def is_valid(self):
        return not self.is_error


# TODO - implement this class using abstract base classes module
class _AVBackend(object):
    # class properties
    name: str = None
    required_modules: Tuple[ModuleType, ...] = None
    required_module_names: Tuple[str, ...] = None

    # magic methods
    def __new__(cls, *args, **kwargs):
        if cls == _AVBackend:
            raise RuntimeError(f'The "{cls.__name__}" class cannot be '
                               f'instantiated directly. You must use a child '
                               f'class.')

        class_props = (cls.name,
                       cls.required_modules,
                       cls.required_module_names)
        if None in class_props:
            cprop_iter = zip(
                ('name', 'required_modules', 'required_module_names'),
                class_props)
            cprop_str = (
                '['
                + ', '.join(f'"{n}"' for n, v in cprop_iter if v is None)
                + ']')
            raise RuntimeError(f'The class properties: {cprop_str} '
                               f'must be defined (i.e. cannot be `None`).')

        return super().__new__(cls)

    def __init__(self, path):
        self._check_for_dependencies()
        self._path = path

    # Abstract Methods
    def _get_num_frames_estimate(self):
        raise NotImplementedError()

    def _get_num_frames(self):
        raise NotImplementedError()

    def _get_frame_rate(self):
        raise NotImplementedError()

    def _get_frame_shape(self):
        raise NotImplementedError()

    def _frame_gen_helper(self, frame_index_list):
        raise NotImplementedError()

    def load_frame(self, frame_index) -> Frame:
        raise NotImplementedError()

    def _frame_to_ndarray_helper(self, frame, pixel_format):
        """
        pixel_format:
          - rgb24 (default; 8-bit RGB)
          - rgb48le (16-bit lower-endian RGB)
          - bgr24 (8-bit BGR; openCVs default colorspace)
          - gray (8-bit grayscale)
          - yuv444p (8-bit channel-first YUV)
        """
        raise NotImplementedError()

    def write_frames(self,
                     frames: np.ndarray,
                     fps: Union[int, float, fractions.Fraction],
                     is_color: bool,
                     codec: Optional[str],
                     **kwargs):
        raise NotImplementedError()

    # Methods
    def _check_for_dependencies(self):
        modules = self.required_modules
        if None in modules:
            msg = self._check_deps_err_msg()
            raise RuntimeError(msg)

    def frame_generator(self, frame_index_list) -> Generator[Frame]:
        if not frame_index_list:
            raise StopIteration

        frame_index_arr = np.array(frame_index_list)
        frame_index_diff = np.diff(frame_index_arr)
        frame_index_err = np.any(frame_index_diff <= 0)
        if frame_index_err:
            msg = ('frame_index_list must be a monotonically increasing list '
                   'of non-negative, integer-valued numbers.')
            raise ValueError(msg)

        yield from self._frame_gen_helper(frame_index_list)

    def frame_to_ndarray(self,
                         frame: Frame,
                         pixel_format: str
                         ) -> np.ndarray:
        if frame.is_error:
            msg = f'An invalid frame cannot be converted to an ndarray; ' \
                  f'passed frame: {frame}. Using {self.name:s} backend.'
            raise ValueError(msg)
        return self._frame_to_ndarray_helper(frame.frame_data, pixel_format)


    @classmethod
    def _check_deps_err_msg(cls):
        missing_package_names = []
        for m, name in zip(cls.required_modules, cls.required_module_names):
            if m is None:
                missing_package_names.append(name)

        package_list_str = '[{:s}]'.format(
            ', '.join(f'"{n}"' for n in missing_package_names))
        msg = (
            'The packages: {:s} were not found; they are required '
            'to use the the "{:s}" backend.').format(
            package_list_str, cls.name)
        return msg

    # Class Computed Properties
    @property
    def path(self) -> str:
        return self._path

    @property
    def num_frames_estimate(self) -> int:
        return self._get_num_frames_estimate()

    @property
    def num_frames(self) -> int:
        return self._get_num_frames()

    @property
    def frame_rate(self) -> fractions.Fraction:
        return self._get_frame_rate()

    @property
    def frame_shape(self) -> Tuple[int, int]:
        return self._get_frame_shape()

    @property
    def _cache_dir(self) -> str:
        return os.path.join(self.path, '..', utilities.CACHE_DIR_NAME)

    @property
    def _filename(self) -> str:
        return os.path.split(self.path)[-1]


class AVIRBackend(_AVBackend):
    name = 'avi-r'
    required_modules = (avi_r, )
    required_module_names = ('avi_r', )

    def __init__(self, path, *, fix_missing):
        super().__init__(path)
        self.fix_missing = fix_missing

    # Class Methods
    @contextlib.contextmanager
    def _avi(self):
        video = avi_r.AVIReader(self.path, fix_missing=self.fix_missing)
        try:
            yield video
        finally:
            video.close()

    # Abstract Method Implementations
    def _get_num_frames_estimate(self):
        with self._avi() as video:
            return video.num_frames

    @csutils.cache
    def _get_num_frames(self):
        count = 0
        with self._avi() as video:
            frame_iterator = video.get_iter()
            for _ in frame_iterator:
                count += 1
        return count

    def _get_frame_shape(self):
        with self._avi() as video:
            return video.shape

    def _get_frame_rate(self):
        with self._avi() as video:
            return fractions.Fraction(video.frame_rate)

    def load_frame(self, frame_index):
        with self._avi() as video:
            frame = video.get_at(frame_index)
            return Frame(frame, is_error=(frame is None))

    def _frame_gen_helper(self, frame_index_list):
        first_frame_index = frame_index_list[0]

        with self._avi() as video:
            video.seek(frame_index_list[0])
            seek_index = first_frame_index
            for frame_index in frame_index_list:
                while seek_index < frame_index:
                    try:
                        video.get()
                    except StopIteration:
                        raise
                    else:
                        seek_index += 1

                try:
                    frame = video.get()
                except StopIteration:
                    raise
                else:
                    seek_index += 1

                yield Frame(frame, is_error=(frame is None))

    def _frame_to_ndarray_helper(self, frame, pixel_format):
        return frame.numpy(format=pixel_format)


class CV2Backend(_AVBackend):
    name = 'cv2'
    required_modules = (cv2, )
    required_module_names = ('cv2', )

    max_search_window_size = 500
    retry_count = 1

    # Class Methods
    @contextlib.contextmanager
    def _capture(self):
        capture = cv2.VideoCapture(self.path)
        try:
            yield capture
        finally:
            capture.release()

    @contextlib.contextmanager
    def _writer(self,
                *,
                fps: int,
                frame_size: Tuple[int, int],
                is_color: Optional[bool] = None,
                codec: str = 'mp4v'):
        if codec.lower() in 'mp4v':
            cv_codec_code = cv2.VideoWriter_fourcc(*codec.lower())
        elif codec.lower() == 'fmp4':
            cv_codec_code = cv2.VideoWriter_fourcc(*codec)
        else:
            msg = (f'The passed codec "{codec}" could not be parsed or '
                   f'utilized to create a `cv2.VideoWriter` object.')
            raise RuntimeError(msg)

        arg_list = (self.path,
                    cv_codec_code,
                    int(fps),
                    frame_size,
                    is_color)
        log.debug('cv2.VideoWriter(*{arg_list})', dict(arg_list=arg_list))
        writer = cv2.VideoWriter(*arg_list)
        try:
            yield writer
        finally:
            writer.release()

    @staticmethod
    def _pixel_format_to_cv_code(pixel_format):
        if pixel_format == 'gray':
            return cv2.COLOR_BGR2GRAY
        else:
            msg = (f'Cannot handle value for `pixel_format` property '
                   f'({pixel_format}).')
            raise RuntimeError(msg)

    @staticmethod
    def _capture_seek_to(cap, pos):
        return int(cap.set(cv2.CAP_PROP_POS_FRAMES, pos))

    @staticmethod
    def _capture_get_pos(cap):
        return int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Abstract Method Implementations
    def _get_num_frames_estimate(self):
        with self._capture() as capture:
            return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def _get_num_frames(self):
        guess = self.num_frames_estimate
        with self._capture() as capture:
            # check below guess by ensuring that the capture can
            # seek to the desired frame
            count = 0
            while True:
                if count > self.max_search_window_size:
                    return None
                self._capture_seek_to(capture, guess)
                cap_pos = self._capture_get_pos(capture)
                if cap_pos == guess:
                    break
                if guess == 0:
                    return None
                guess -= 1
                count += 1

            # check above guess by ensuring the capture canNOT seek to
            # a subsequent frame
            count = 0
            while True:
                if count > self.max_search_window_size:
                    return None
                self._capture_seek_to(capture, guess + 1)
                cap_pos = self._capture_get_pos(capture)
                if cap_pos == guess:
                    # that is, if seek_to fails (as it should at the last
                    # frame)
                    break
                if cap_pos == (guess + 1):
                    # that is, if seek_to succeeds (meaning we're really not
                    # at the last frame)
                    guess += 1
                else:
                    # something unexpected occurred
                    return None
                count += 1

            return guess

    def _get_frame_rate(self):
        with self._capture() as capture:
            return fractions.Fraction(capture.get(cv2.CAP_PROP_FPS))

    def _get_frame_shape(self):
        with self._capture() as capture:
            return (int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))

    def _frame_gen_helper(self, frame_index_list):
        with self._capture() as capture:
            self._capture_seek_to(capture, frame_index_list[0])

            for frame_index in frame_index_list:
                cap_pos = self._capture_get_pos(capture)
                old_cap_pos = None
                while cap_pos < frame_index:
                    if old_cap_pos is None:
                        capture.read()
                    elif cap_pos == old_cap_pos:
                        # that is, if `capture.read()` has no effect
                        break
                    else:
                        capture.read()
                    old_cap_pos = cap_pos
                    cap_pos = self._capture_get_pos(capture)

                is_valid, frame = False, None
                for i in range(self.retry_count + 1):
                    cap_pos = self._capture_get_pos(capture)
                    if frame_index != cap_pos:
                        if (i > 0) and (i == self.retry_count):
                            # no need to run the seek_to function
                            # on the final retry iteration
                            err = RuntimeError(
                                f'Could not seek to desired frame index '
                                f'({frame_index}); capture position is at '
                                f'{cap_pos}. Failed to get frame.')
                            break
                        self._capture_seek_to(capture, frame_index)
                        continue
                    is_valid, frame = capture.read()
                    err = None
                    break

                yield Frame(frame,
                            is_error=(not is_valid),
                            index=frame_index,
                            exc_info=err)

    def load_frame(self, frame_index):
        with self._capture() as capture:
            if not frame_index == 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            cap_pos = self._capture_get_pos(capture)
            if frame_index != cap_pos:
                err = RuntimeError(
                    f'Desired frame index ({frame_index}) did not match '
                    f'capture position ({cap_pos}). Failed to get frame.')
                return Frame(None,
                             is_error=True,
                             index=frame_index,
                             exc_info=err)

            is_valid, frame = capture.read()
            return Frame(frame, is_error=(not is_valid), index=frame_index)

    def _frame_to_ndarray_helper(self, frame, pixel_format):
        code = self._pixel_format_to_cv_code(pixel_format)
        return cv2.cvtColor(frame, code)

    def write_frames(self,
                     frames: np.ndarray,
                     fps: Union[int, float, fractions.Fraction],
                     is_color: bool,
                     codec: Optional[str] = 'mp4v',
                     **kwargs):

        if is_color:
            assert frames.ndim == 4, 'Color video ndarray should have 4 dims.'
            num_frames, height, width, num_channels = frames.shape
        else:
            assert frames.ndim == 3, \
                'Greyscale video ndarray should have 3 dims.'
            num_frames, height, width = frames.shape
            num_channels = 1

        if codec is not None:
            if codec.lower() in ('mp4v', 'fmp4'):
                if not self.path.endswith('.mp4'):
                    msg = (f'To write a .mp4 file, the file path must end with '
                           f'".mp4", working video path is "{self.path}".')
                    raise RuntimeError(msg)
            else:
                log.warning('May not be able to write video with passed codec '
                            '"{}"; using "{self.name}" backend.',
                            codec, self.name)

        fps_int = int(fps)

        log.debug('Writing {:d}{:s} frames having dtype {}, and size '
                  '(w: {:d}, h: {:d}) '
                  'to a {:d} fps video file with codec {:s} at path "{:s}".',
                  num_frames,
                  ' color' if is_color else ' greyscale',
                  frames.dtype,
                  width,
                  height,
                  fps_int,
                  codec or '[unknown]',
                  self.path)

        with self._writer(fps=fps_int,
                          frame_size=(width, height),
                          is_color=is_color,
                          codec=codec,
                          **kwargs) as writer:
            for i, frame in enumerate(frames):
                writer.write(frame)


class PIMSBackend(_AVBackend):
    name = 'pims'
    required_modules = (pims, av, )
    required_module_names = ('pims', 'av', )

    def __init__(self, path):
        super().__init__(path)
        self._pims_idx_reader = None

    def _init_pims_reader(self):
        csutils.touchdir(self._cache_dir)
        toc_cache = None
        if os.path.exists(self._pyav_cache_path):
            try:
                log.debug('Attempting to load cached toc at "{:s}".',
                          self._pyav_cache_path)
                toc_cache = csutils.load_from_disk(self._pyav_cache_path)
            except Exception as e:
                log.debug('Error encountered when loading from cache:',
                          exc_info=e)

        log.debug('Initializing `pims.PyAVReaderIndexed` object '
                  'for pims backend.')
        if toc_cache is not None:
            log.debug('Using cached `toc`.')

        self._pims_idx_reader = pims.PyAVReaderIndexed(self.path,
                                                       toc=toc_cache)
        if toc_cache is None:
            csutils.save_to_disk(self.pyav_reader.toc, self._pyav_cache_path)

    @contextlib.contextmanager
    def _av_container(self):
        with av.open(self.path) as container:
            yield container

    @staticmethod
    def _get_container_video_stream(container):
        return container.streams.video[0]

    # Abstract Method Implementations
    def _get_num_frames_estimate(self):
        if self._pyav_reader_initialized:
            return self._get_num_frames()
        else:
            with self._av_container() as container:
                v_stream = self._get_container_video_stream(container)
                return int(v_stream.frames)

    def _get_num_frames(self):
        return len(self.pyav_reader)

    def _get_frame_rate(self):
        with self._av_container() as container:
            v_stream = self._get_container_video_stream(container)
            codec = v_stream.codec_context
            if codec.framerate is not None:
                return codec.framerate

            if v_stream.base_rate is not None:
                return v_stream.base_rate

            if v_stream.average_rate is not None:
                return v_stream.average_rate

            raise RuntimeError(f'Could not determine frame rate; using '
                               f'{self.name} backend.')

    def _get_frame_shape(self):
        return self.pyav_reader.frame_shape[:2]

    def _frame_gen_helper(self, frame_index_list) -> Frame:
        for frame_index in frame_index_list:
            yield self.load_frame(frame_index)

    def load_frame(self, frame_index) -> Frame:
        frame_kwargs = dict(index=frame_index, is_error=True)
        try:
            frame_data = self.pyav_reader[frame_index]
        except IndexError:
            raise
        except RuntimeError as e:
            frame_data = None
            frame_kwargs.update(exc_info=e)
            log.debug('`load_frame` at index {:d} failed with {:s} backend.',
                      frame_index, self.name, exc_info=e)
        else:
            frame_kwargs.update(is_error=False)
        return Frame(frame_data, **frame_kwargs)

    def _frame_to_ndarray_helper(self, frame, pixel_format):
        """
        pixel_format:
          - rgb24 (default; 8-bit RGB)
          - rgb48le (16-bit lower-endian RGB)
          - bgr24 (8-bit BGR; openCVs default colorspace)
          - gray (8-bit grayscale)
          - yuv444p (8-bit channel-first YUV)
        """
        if pixel_format.lower() == 'gray':
            # FIXME - is this line correct? Should it be the 1st color channel?
            return frame[:, :, 1]
        else:
            raise NotImplementedError()

    # properties
    @property
    def pyav_reader(self):
        if not self._pyav_reader_initialized:
            self._init_pims_reader()
        return self._pims_idx_reader

    @property
    def _pyav_reader_initialized(self):
        return self._pims_idx_reader is not None

    @property
    def _pyav_cache_path(self):
        return os.path.join(self._cache_dir,
                            f'pyav_reader({self._filename}).pkl')


class AVDataset(object):
    size_warning_cutoff_gb = 0.5
    valid_backends = (AVIRBackend, CV2Backend, PIMSBackend)

    def __init__(self,
                 path,
                 *,
                 name=None,
                 fps=None,
                 lazy=True,
                 pixel_format='gray',
                 pixel_pitch_um=None,
                 backend='pims'):
        """
        pixel_format:
          - rgb24 (default; 8-bit RGB)
          - rgb48le (16-bit lower-endian RGB)
          - bgr24 (8-bit BGR; openCVs default colorspace)
          - gray (8-bit grayscale)
          - yuv444p (8-bit channel-first YUV)
        """

        # first, initialize properties handled elsewhere
        self._backend_obj = None
        self._backend_class = None
        self._image_ndarray = None
        self._is_valid_frame = None
        self._num_frames = None
        self._fix_missing = None

        # then, set path
        self.path = path

        # then, initialize other instance attributes
        self.name = name
        log.info('Creating an avi dataset ({}).', self.name or '[unnamed]')

        # initialize backend
        self.backend = backend

        self.pixel_format = pixel_format
        self.pixel_pitch_um = pixel_pitch_um
        self._frame_rate_hz = fps
        log.debug('Set AVDataset frame rate to {}',  self.frame_rate_hz)

        self.fix_missing = False
        self.stride = 1

        if not lazy:
            self.load_all_images()

    def get_frames(self,
                   start_frame: int,
                   end_frame: int,
                   *,
                   interactive: bool = False,
                   allow_failed_frames: bool = False,
                   use_accurate_frame_count: bool = False,
                   remove_trailing_invalid_frames: bool = False,
                   _silent: bool = False
                   ) -> Tuple[np.ndarray, np.ndarray]:

        err_msg_fmt = ('Passed index for `%s` (%d) is outside of the valid '
                       'range [0, %d%s.')

        if use_accurate_frame_count or (self._backend_class is PIMSBackend):
            est_num_frames = self.num_file_frames
        else:
            est_num_frames = self.num_file_frames_estimate

        if not (0 <= start_frame < est_num_frames):
            msg = err_msg_fmt % ('start_frame', start_frame,
                                 est_num_frames, ')')
            raise ValueError(msg)
        if not (0 <= end_frame <= est_num_frames):
            msg = err_msg_fmt % ('end_frame', end_frame,
                                 est_num_frames, ']')
            raise ValueError(msg)

        if end_frame <= start_frame:
            raise ValueError('`start_frame` (%d) cannot be greater than or '
                             'equal to `end_frame` (%d).'
                             % (start_frame, end_frame))

        target_num_frames = (end_frame - start_frame) // self.stride

        if not _silent:
            log.debug('Getting the first frame of the range.')

        first_frame = self._backend.load_frame(start_frame)
        if first_frame.is_error:
            raise RuntimeError('Could not obtain the first frame for '
                               'building a load spec.')
        first_frame_ndarray = self._backend.frame_to_ndarray(
            first_frame,
            pixel_format=self.pixel_format)

        if not _silent:
            log.debug('Frame has datatype: {}.', first_frame_ndarray.dtype)

        if interactive:
            data_size = (first_frame_ndarray.size
                         * first_frame_ndarray.itemsize
                         * target_num_frames)
            data_size_gb = data_size / 1E9
            if data_size_gb > self.size_warning_cutoff_gb:
                log.warning('You are attempting to load at least {:.1f} GB '
                            'into memory from av file at "{:s}" (frames {:d} '
                            'to {:d}. This will take some time.\n'
                            'Do you want to proceed with (f)ull, '
                            '(P)artial, or (n)o loading of the data? [P/f/n]',
                            data_size_gb, self.path, start_frame, end_frame-1)
                response = input('\n')
                if response.lower() == 'f':
                    pass
                elif response.lower() == 'n':
                    log.warning('Exiting execution of load_frames().')
                    return None
                else:
                    log.info('Performing a partial load.')
                    log.warning('What percent of the av movie would you like '
                                'to load? [0 - 100] (Just press enter to load '
                                'approx 1 GB of data.)')
                    try:
                        percent_frames_to_load = float(input('\n'))
                    except Exception as e:
                        log.debug('Error caught from user input.', exc_info=e)
                        target_gb = 1
                        num_frames_to_load = (
                                target_gb * target_num_frames
                                // round(data_size_gb))
                    else:
                        num_frames_to_load = round(
                            target_num_frames * (percent_frames_to_load / 100))

                    if not _silent:
                        log.info("Loading {} out of {} frames.",
                                 num_frames_to_load, target_num_frames)
                    target_num_frames = num_frames_to_load

        target_end_frame = start_frame + target_num_frames
        frame_index_iterator = range(start_frame,
                                     target_end_frame,
                                     self.stride)
        output_num_frames = len(frame_index_iterator)
        output_shape = (output_num_frames,) + self.frame_shape

        if not _silent:
            log.debug('Initializing the numpy array.')

        output_image_ndarray = np.zeros_like(first_frame_ndarray,
                                             shape=output_shape)
        output_frame_valid = np.zeros(output_shape[:1], dtype=bool)

        _counter = output_num_frames // 20
        if not _silent:
            log.info('Loading {:d} av frames to a numpy array.',
                     output_num_frames)
            log.debug('Output array will have shape {}.', output_shape)

        fail_msg = 'Failed to grab frame at index {:d}.'

        frame_generator = self._backend.frame_generator(frame_index_iterator)
        indexed_frame_iterator = enumerate(zip(frame_index_iterator,
                                               frame_generator))

        start = time.time()
        if not _silent:
            log.debug('Reading in image file with {} backend.',
                      self.backend)

        for i, (frame_index, frame) in indexed_frame_iterator:
            if not _silent:
                if (i > 0) and (i % _counter == 0):
                    log.debug('Loading frame {:9d} of {:9d} ({:.0f}% '
                              'complete).',
                              (i + 1), output_num_frames,
                              100 * ((i + 1) / output_num_frames))

            if frame.is_error:
                if not allow_failed_frames:
                    msg = fail_msg.format(frame_index) + f'frame: {frame}.'
                    raise RuntimeError(msg)
                log.debug('{:s} Marking frame as invalid.',
                          fail_msg.format(frame_index),
                          frame,
                          exc_info=frame.exc_info)
                continue

            frame_ndarray = self._backend.frame_to_ndarray(
                frame,
                pixel_format=self.pixel_format)
            output_frame_valid[i] = True
            output_image_ndarray[i, :, :] = frame_ndarray

        if not _silent:
            csutils.log_time_delta(log, start, 'Loading av frames')

        if remove_trailing_invalid_frames:
            last_valid_index = np.where(output_frame_valid)[0][-1]
            last_index = len(output_frame_valid) - 1
            log.debug('Last valid index for output is ({}), '
                      'and last index is ({}).',
                      last_valid_index, last_index)
            if last_valid_index < last_index:
                removing_last = last_index - last_valid_index
                log.info('Removing last {} frames. Truncating output arrays '
                         'to the last valid index.', removing_last)
                output_slice = np.s_[0:(last_valid_index + 1)]
                output_image_ndarray = output_image_ndarray[output_slice]
                output_frame_valid = output_frame_valid[output_slice]

        return output_image_ndarray, output_frame_valid

    def load_all_images(self, force=False):
        if (not force) and (self._image_ndarray is not None):
            log.debug('All images already loaded.')
            return

        self._image_ndarray, self._is_valid_frame = self.get_frames(
            start_frame=0,
            end_frame=self.num_file_frames,
            interactive=True,
            allow_failed_frames=True,
            remove_trailing_invalid_frames=True)
        self._num_frames = len(self._image_ndarray)

        log.info('Loading all av images complete.')

    # properties
    @property
    def video_length_second(self):
        return self.frame_period_second * self.num_frames

    @property
    def num_file_frames_estimate(self):
        return self._backend.num_frames_estimate

    @property
    def num_file_frames(self):
        return self._backend.num_frames

    @property
    def frame_rate_hz(self):
        if self._frame_rate_hz is None:
            return self._backend.frame_rate
        else:
            return self._frame_rate_hz

    @property
    def num_frames(self):
        if self._num_frames is None:
            return self._backend.num_frames // self.stride
        else:
            return self._num_frames

    @property
    def shape(self):
        return (self.num_frames, ) + self.frame_shape

    @property
    def frame_shape(self):
        return self._backend.frame_shape

    @property
    def frame_period_second(self):
        return 1 / self.frame_rate_hz

    @property
    def image_ndarray(self):
        self.load_all_images()
        return self._image_ndarray

    @property
    def is_valid_frame(self):
        self.load_all_images()
        return self._is_valid_frame

    @property
    def _cache_dir(self):
        return os.path.join(self.path, '..', CACHE_DIR_NAME)

    @property
    def _filename(self):
        return os.path.split(self.path)[1]

    @property
    def fix_missing(self):
        return self._fix_missing

    @fix_missing.setter
    def fix_missing(self, val):
        self._fix_missing = val
        if self._backend_class == AVIRBackend:
            self._backend.fix_missing = val

    @property
    def _backend(self) -> _AVBackend:
        if self._backend_obj is None:
            log.debug('Initializing backend object {:s}("{:s}")',
                      self._backend_class.__name__, self.path)
            init_kwargs = dict()
            if self._backend_class is AVIRBackend:
                init_kwargs['fix_missing'] = self.fix_missing
            self._backend_obj = self._backend_class(self.path, **init_kwargs)
        return self._backend_obj

    @property
    def backend(self) -> str:
        return self._backend_class.name.lower()

    @backend.setter
    def backend(self, val: str):
        val_lower = val.lower()
        if (self._backend_class is not None) and (self.backend == val_lower):
            # no update needed
            return

        backend_class = [c for c in self.valid_backends
                         if val_lower == c.name.lower()]

        if not backend_class:
            msg = f'The passed backend name "{val}" is not a valid backend.'
            raise ValueError(msg)

        if len(backend_class) > 1:
            raise RuntimeError(f'"{val}" yielded more than one valid backend.')

        self._backend_obj = None
        self._backend_class = backend_class[0]

    # class methods
    @classmethod
    def from_dataset_info(cls,
                          *,
                          dataset_info: ZMIADatasetInfo,
                          **kwargs
                          ) -> AVDataset:
        video_path = dataset_info.full_path
        kwargs.setdefault('name', dataset_info.name)
        return cls(video_path, **kwargs)


# deprecated class, replaced by AVDataset
class AVIDataset(AVDataset):
    def __init__(self, *args, **kwargs):
        log.warning('The "{}" class is deprecated; '
                    'use "{}" instead.',
                    self.__class__.__name__,
                    AVDataset.__name__)
        super().__init__(*args, **kwargs)


def convert_avi_to_mp4(avi_file_path, output_name=None):
    path_head, path_tail = os.path.split(avi_file_path)
    if output_name is None:
        output_name = '.'.join(path_tail.split('.')[:-1])

    output_full_path = os.path.join(path_head, '{}.mp4'.format(output_name))
    os.popen('ffmpeg -i "{input}" -ac 2 -b:v 2000k -c:a aac -c:v libx264 '
             '-b:a 160k -vprofile high -bf 0 -strict experimental '
             '-f mp4 "{output}"'
             .format(input=avi_file_path, output=output_full_path))

    return output_full_path


valid_writer_backends: Tuple[Type[_AVBackend], ...] = (CV2Backend, )
default_codecs: Dict[str, str] = {
    'mp4': 'mp4v',
}


def write_array_to_video_file(
        video_data: np.ndarray,
        path: str,
        *,
        frame_rate_hz: Union[int, float, fractions.Fraction],
        backend: str = 'cv2',
        file_format: str = 'mp4',
        codec: Optional[str] = None):

    # determine the backend class to use
    backend_lc = backend.lower()
    backend_class_match = []
    for bkd_cls in valid_writer_backends:
        if bkd_cls.name.lower() == backend_lc:
            backend_class_match.append(bkd_cls)

    if not backend_class_match:
        msg = (f'The passed backend "{backend}" is not valid for writing video '
               f'files.')
        raise ValueError(msg)

    if len(backend_class_match) > 1:
        msg = (f'The passed backend "{backend}" matched more than one valid ' 
               f'backend class.')
        raise RuntimeError(msg)

    log.debug('Using backend "{}" for writing ndarray to file.', backend)
    backend_class: Type[_AVBackend] = backend_class_match[0]

    # create backend object for performing frame writing
    backend_obj = backend_class(path=path)

    # calculate if working with color image
    ndim = video_data.ndim
    if ndim == 3:
        is_color = False
    elif ndim == 4:
        is_color = True
    else:
        msg = (f'Expected video data to have either 3 dimensions (greyscale) '
               f'or 4 dimensions (color); passed video data has {ndim} '
               f'dimensions.')
        raise ValueError(msg)

    # determine codec
    if codec is None:
        try:
            codec = default_codecs[file_format]
        except KeyError:
            msg = ('There is not a default codec associated with the '
                   '"{}" file format. Pass an argument to the'
                   '`codec` parameter directly, or use a different file '
                   'format. Backend will attempt to find a codec '
                   'automatically.')
            log.warning(msg, file_format)

    # check passed frame rate property
    if not isinstance(frame_rate_hz, (int, float, fractions.Fraction)):
        msg = (f'`frame_rate_hz` parameter must be a numeric value; '
               f'got a value of type {frame_rate_hz.__class__.__name__}.')
        raise TypeError(msg)

    if not frame_rate_hz > 0:
        msg = '`frame_rate_hz` parameter must be greater than zero.'
        raise ValueError(msg)

    # call backend writer
    log.info('Writing frames to {} video file at "{}".', file_format, path)
    backend_obj.write_frames(frames=video_data,
                             fps=frame_rate_hz,
                             is_color=is_color,
                             codec=codec)
    log.info('Writing of frames complete.')


PDS_CONV_ENV_NAME = 'PDS_CONV_EXE'
PDS_CONV_DIR_FMR = '{:s}_CONVERTED'
PDS_CONV_FAILED_FMT = '{:s}_(FAILED)'
PDS_CONV_AVI_GLOB = '*.avi'
PDS_CONV_TIME_TABLE_GLOB = '*TIME.csv'

try:
    PXL_CONVERSION_EXE_PATH: Optional[str] = os.environ[PDS_CONV_ENV_NAME]
except KeyError:
    PXL_CONVERSION_EXE_PATH = None


class PDSConversionProc(object):
    def __init__(self,
                 process: Optional[subprocess.Popen],
                 video_path: str,
                 video_name: str,
                 output_dir: str,
                 log_fd: Optional[int],
                 is_running: bool,
                 number: int):
        self.process: Optional[subprocess.Popen] = process
        self.video_path: str = video_path
        self.video_name: str = video_name
        self.output_dir: str = output_dir
        self.log_fd: Optional[int] = log_fd
        self.is_running: bool = is_running
        self.number: int = number
        self.did_fail: Optional[bool] = None


class PDSConversionResult(NamedTuple):
    output_dir: str
    did_fail: bool


def convert_pixelink_datastream_to_avi(
        pds_paths: List[os.PathLike | str | bytes],
        conversion_exe_path: Optional[os.PathLike | str | bytes] = None,
        lazy: bool = True,
        reattempt_failed: bool = False,
        ) -> List[PDSConversionResult]:

    _conversion_exe_path = conversion_exe_path or PXL_CONVERSION_EXE_PATH
    if _conversion_exe_path is None:
        msg = (f'No pds to avi conversion executable could be found. '
               f'Check the value for environment variable '
               f'"{PDS_CONV_ENV_NAME}" or pass the executable path directly.')
        raise RuntimeError(msg)

    log.info("Using conversion executable at: {:s}",
             _conversion_exe_path)

    log.info('Performing conversion on {:d} files.',
             len(pds_paths))

    processes: List[PDSConversionProc] = []

    i_path: int
    video_path: str
    for i_path, video_path in enumerate(pds_paths):
        video_basedir, video_filename = os.path.split(video_path)
        video_name, _ = os.path.splitext(video_filename)

        output_dir = os.path.join(
            video_basedir,
            PDS_CONV_DIR_FMR.format(video_name))

        failed_dir = PDS_CONV_FAILED_FMT.format(output_dir)
        has_already_failed = os.path.isdir(failed_dir)
        has_already_converted = os.path.isdir(output_dir) or has_already_failed

        will_run: bool = True

        if lazy and has_already_converted and (not has_already_failed):
            log.info('Video {:02d} ({:s}) already converted, skipping.',
                     i_path, video_name)
            will_run = False

        if (lazy and has_already_converted and has_already_failed and
                (not reattempt_failed)):
            log.info('Video {:02d} ({:s}) already converted & failed, '
                     'skipping.', i_path, video_name)
            output_dir = failed_dir
            will_run = False
        elif (lazy and has_already_converted and has_already_failed and
                reattempt_failed):
            log.info('Reattempting conversion of video {:02d} ({:s}) which '
                     'failed in a former conversion run.',
                     i_path, video_name)

        if will_run:
            log.info('Beginning subprocess to convert video {:02d},\n\t"{:s}".',
                     i_path, video_path)
            if has_already_failed:
                log.info('Removing old failed conversion directory for video '
                         '{:02d}.', i_path)
                shutil.rmtree(failed_dir)

            log_path = os.path.join(
                output_dir,
                f'log_{dt.datetime.now().strftime("%Y%m%d-%H%M%S")!s}.log')

            csutils.touchdir(output_dir)
            log_fd = os.open(log_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)

            popen = subprocess.Popen(args=[_conversion_exe_path, video_path],
                                     stdout=log_fd,
                                     stderr=log_fd)
        else:
            popen = None
            log_fd = None

        proc_obj = PDSConversionProc(
            popen, video_path, video_name, output_dir, log_fd,
            is_running=will_run,
            number=i_path)

        if has_already_converted:
            proc_obj.did_fail = has_already_failed

        processes.append(proc_obj)

    try:
        while processes and any([p.is_running for p in processes]):
            for p in processes:
                if not p.is_running:
                    continue

                exit_code = p.process.poll()
                if exit_code is None:
                    continue

                p.is_running = False
                p.did_fail = exit_code != 0

                if p.did_fail:
                    log.warning('Video {:02d} ({:s}) conversion failed with '
                                'exit code {:d} .',
                                p.number, p.video_name, exit_code)

                    os.close(p.log_fd)

                    log.info('Marking directory for video {:02d} ({:s}) '
                             'as failed.',
                             p.number, p.video_name)
                    new_output_dir = PDS_CONV_FAILED_FMT.format(p.output_dir)
                    shutil.move(p.output_dir, new_output_dir)
                    p.output_dir = new_output_dir
                    continue

                log.info('Video {:02d} ({:s}) conversion succeeded!',
                         p.number, p.video_name)

            time.sleep(1)

    except Exception as exc:
        log.error('An error occurred during distributed '
                  'subprocesses execution.',
                  exc_info=exc)
        for p in processes:
            if p.process.poll() is None:
                log.warning('Killing process {:02d}.', p.number)
                p.process.kill()
        raise

    finally:
        for p in processes:
            if p.log_fd is None:
                continue
            try:
                os.close(p.log_fd)
            except OSError:
                pass

    results = [PDSConversionResult(p.output_dir, p.did_fail)
               for p in processes]
    return results

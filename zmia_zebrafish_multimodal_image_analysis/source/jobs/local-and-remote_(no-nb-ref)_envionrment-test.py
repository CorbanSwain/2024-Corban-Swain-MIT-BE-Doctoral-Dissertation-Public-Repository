#!python3
# local-and-remote_(no-nb-ref)_envionrment-test.py

# IMPORTS
# - standard imports
import os
import cProfile
import pstats
import sys

# - local imports
sources_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..'))
if sources_dir not in sys.path:
    sys.path.append(sources_dir)

from utilities import *
import video_utils
import c_swain_python_utils as csutils
import prairie_view_imports as pvi
import imaging_dataset as imd

# - additional imports
import warnings
import numpy
import skimage
from skimage import data
from qtpy.QtCore import QTimer
import time
import h5py
import pandas
import scipy
import matplotlib
import seaborn

try:
    import napari
except Exception as e:
    warnings.warn(f'`napari` module import failed: {e}')
    napari = None
else:
    import napari._qt

try:
    import caiman as cm
    from caiman.utils.utils import download_demo
except Exception as e:
    warnings.warn(f'`caiman` module import failed: {e}')
    cm = None

try:
    import ants
except Exception as e:
    warnings.warn(f'`ants` module import failed: {e}.')
    ants = None

try:
    import ipynbname
except Exception as e:
    warnings.warn(f'`ipynbname` module import failed: {e}.')
    ipynbname = None

try:
    import zebrazoom
except Exception as e:
    warnings.warn(f'`zebrazoom` module import failed: {e}.')
    zebrazoom = None

# logger
log = csutils.get_logger(__name__)

# standard global variables
jobs_dir = os.path.split(__file__)[0]
sources_path = os.path.abspath(os.path.join(jobs_dir, '..'))
repo_dir = os.path.abspath(os.path.join(sources_path, '..'))
config_dir = os.path.abspath(os.path.join(repo_dir, 'configs',))
debug_mode = False

# custom global variables


# Main Script ##################################################################
def main():
    log.info("This is a test script.")

    log.info('Checking on semi-optional packages:')
    msg = 'ipynbname = {}'.format(ipynbname)
    if ipynbname is None:
        log.warning('`ipynbname` module not loaded! ({})', msg)
    else:
        log.info(msg)

    msg = 'zebrazoom = {}'.format(zebrazoom)
    if zebrazoom is None:
        log.warning('`zebrazoom` module not loaded! ({})', msg)
    else:
        log.info(msg)

    log.info('napari = {}', napari)
    if napari is not None:
        try:
            log.info(" > Testing napari gui viewer.")
            viewer = napari.Viewer()
            viewer.add_image(data.cell(), name='cell')
            viewer.close()
        except Exception as _e:
            log.warning(f' > `napari` viewer test failed with error: {e}',
                        exc_info=_e)
        else:
            log.info(' > `napari` viewer test passed.')
    else:
        log.warning(' > `napari` module not loaded!')

    log.info('caiman = {}', cm)
    if cm is not None:
        try:
            log.info(" > Testing caiman movie player.")
            movie_path = download_demo('Sue_2x_3000_40_-46.tif')
            log.info(f" > Original movie for demo is in {movie_path}")
            movie_orig = cm.load(movie_path)
            downsampling_ratio = 0.2  # subsample 5x
            log.info(' > Press q to quit the movie player.')
            (movie_orig
             .resize(fz=downsampling_ratio)
             .play(gain=1.3,
                   q_max=99.5,
                   fr=30,
                   plot_text=True,
                   magnification=2,
                   do_loop=False,
                   backend='opencv'))
        except Exception as _e:
            log.warning(f' > `caiman` movie player failed with error',
                        exc_info=_e)
        else:
            log.info(' > `caiman` movie player test passed.')
    else:
        log.warning(' > `caiman` module not loaded!')

    log.info('ants = {}', ants)
    try:
        _ = ants.from_numpy
    except Exception:
        log.warning(' > `ants` module import failed.')
    else:
        log.info(' > `ants` module installed and imported successfully.')

    log.info('Checking on av packages:')
    log.info('video_utils.av = {}', video_utils.av)
    log.info('video_utils.avi_r = {}', video_utils.avi_r)
    log.info('video_utils.pims = {}', video_utils.pims)
    log.info('video_utils.cv2 = {}', video_utils.cv2)

    log.info("Done with tests.")
################################################################################


# custom functions
def foo():
    pass


# handle running of the job
if __name__ == '__main__':
    # get jobfile name without extension
    _file = csutils.no_ext_basename(__file__)

    # setup logging    
    log_dir = os.path.join(repo_dir, 'logs')
    csutils.touchdir(log_dir) 
    log_path = os.path.join(log_dir, f'{_file}.log')
    csutils.apply_standard_logging_config(
        file_path=log_path,
        window_level='debug' if debug_mode else 'info',
        window_format='debug' if debug_mode else 'default')

    # run job and catch exceptions
    try:
        log.info('v' * 80)
        log.info('Beginning Job `{}`.', _file)
        main()
    except KeyboardInterrupt:
        log.warning('Job interrupted, exiting.')
    except Exception as e:
        log.critical('Job Failed.', exc_info=e)
        raise e
    else:
        log.info('Job Completed')
    finally:
        log.info('^' * 80)


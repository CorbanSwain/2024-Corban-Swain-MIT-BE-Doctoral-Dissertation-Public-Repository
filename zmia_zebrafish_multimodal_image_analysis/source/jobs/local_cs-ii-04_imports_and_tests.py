#!python3
# local_cs-ii-04_imports_and_tests.py

# imports
import os.path
import skimage.io
import pprint as pp
import c_swain_python_utils as csutils
import prairie_view_imports as pvi
import napari
import numpy as np

# logger
log = csutils.get_logger(__name__)

# standard global variables
sources_path = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..'))
config_dir = os.path.abspath(os.path.join(sources_path, '..', 'configs',))
debug_mode = True

# custom global variables


# Main Script ##################################################################
def main():
    # settings
    config_filename = 'cs-ii-07_test_config_1.yml'

    # setting config path
    config_path = os.path.join(config_dir, config_filename)
    log.info('Set PrarieView import configuration path to "{}"', config_path)
    config = pvi.set_config(config_path)

    # loading in one dataset
    live_dataset = pvi.load_dataset(0, config=config, lazy=False)

    # logging a few pages of the data set
    log.debug(pp.pformat(live_dataset.I[:3]))

    # viewing the dataset with napari
    viewer = napari.Viewer(axis_labels=live_dataset.dimensions)
    viewer.add_image(data=live_dataset.I[1:, 1:],
                     scale=np.array(live_dataset.voxel_pitch_um),
                     contrast_limits=[0, live_dataset.int_max],
                     name=live_dataset.name)
    log.info('Opening a napari session.')
    napari.run()
    log.info('Napari session ended.')
################################################################################


# custom functions


# handle running of the job
if __name__ == '__main__':
    # get jobfile name without extension
    _file = csutils.no_ext_basename(__file__)

    # setup logging
    log_path = os.path.join(sources_path, '..', 'logs', f'{_file}.log')
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


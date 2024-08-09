#!python3
# local_cs-ii-04_multi-track_import_development.py

# imports
import os.path

import c_swain_python_utils as csutils
import prairie_view_imports as pvi
import napari
import numpy as np


# logger
log = csutils.get_logger(__name__)

# standard global variables
sources_path = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..'))
config_dir = os.path.abspath(os.path.join(sources_path, '..', 'configs',))
debug_mode = False

# custom global variables


# Main Script ##################################################################
def main():
    config_path = os.path.join(config_dir, 'cs-ii-07_config.yml')
    pvi_config = csutils.read_yaml(config_path)

    dataset = pvi.load_dataset(1,
                               config=pvi_config,
                               ignore_cache=False,
                               lazy=False)

    pvi.print_misc_dataset_attributes(dataset)

    viewer = napari.Viewer(order=tuple(range(len(dataset.shape))))
    viewer.add_image(
        data=dataset.I,
        scale=dataset.voxel_pitch_um,
        contrast_limits=np.percentile(dataset.I, [1e-2, 100-1e-3]),
        name=dataset.name,
        colormap='magma')

    log.info('Opening a napari session.')
    viewer.dims.axis_labels = dataset.dimensions
    viewer.dims.order = tuple(range(len(dataset.shape)))
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


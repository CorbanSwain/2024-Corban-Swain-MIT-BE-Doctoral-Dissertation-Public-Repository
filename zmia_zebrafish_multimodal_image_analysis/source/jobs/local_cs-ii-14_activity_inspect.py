#!python3
# _JOB_TEMPLATE_(run-location)_(no-nb-ref)_(job_description).py

# imports
import os.path
import c_swain_python_utils as csutils
import prairie_view_imports as pvi
import imaging_dataset as imd
import napari
from utilities import *
import numpy as np
import scipy.ndimage as scim

# logger
log = csutils.get_logger(__name__)

# standard global variables
sources_path = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..'))
config_dir = os.path.abspath(os.path.join(sources_path, '..', 'configs',))
debug_mode = False

# custom global variables


# Main Script ##################################################################
def main():
    # small matter of programing
    config_path = os.path.join(config_dir, 'cs-ii-14_config_01.yml')
    config = ZMIAConfig(config_path)

    dataset_name = 'single-plane-vid-05'
    dataset_info = config.get_dataset_info(name=dataset_name)

    # pvi.load_dataset(dataset_info=dataset_info, clear_cache_and_exit=True)

    pv_dataset = pvi.load_dataset(dataset_info=dataset_info)
    im_dataset = imd.ImagingDataset.from_pvdataset(pv_dataset)

    log.info('Processing imaging data.')

    log.info('Converting to float.')
    im = im_dataset._image_ndarray.astype(float)

    log.info('Converting to F memory storage.')
    im = np.asfortranarray(im)

    log.info('Normalizing.')
    im = im / np.max(im)

    log.info('Applying Blur.')
    im = scim.gaussian_filter(im, sigma=[1, 4, 4])

    log.info('Computing Baseline.')
    baseline_f = np.median(im, axis=0)

    log.info('Computing Delta F.')
    delta_f = im - baseline_f
    delta_f[delta_f < 0] = 0

    log.info('Normalizing delta f into relative change.')
    delta_f_over_f = delta_f / baseline_f

    log.info('Filtering out small values.')
    zero_filter = baseline_f < 2e-3
    delta_f_over_f[:, zero_filter] = 0
    delta_f[:, zero_filter] = 0
    baseline_f[zero_filter] = 0

    viewer = napari.Viewer()
    viewer.add_image(
        data=im_dataset._image_ndarray,
        name=im_dataset.name,
        colormap='magma')

    average_projection = np.max(im_dataset._image_ndarray, axis=0)

    viewer.add_image(
        data=average_projection,
        name=f'MAX-{im_dataset.name}',
        colormap='magma')

    viewer.add_image(
        data=baseline_f,
        name=f'BASELINE-F-{im_dataset.name}',
        colormap='magma')

    viewer.add_image(
        data=delta_f,
        name=f'DF-{im_dataset.name}',
        colormap='magma')

    viewer.add_image(
        data=delta_f_over_f,
        name=f'DF-OVER-F-{im_dataset.name}',
        colormap='magma')

    log.info('Opening a napari session.')
    viewer.dims.axis_labels = im_dataset.dimensions
    viewer.dims.order = tuple(range(len(im_dataset.shape)))
    napari.run()
    log.info('Napari session ended.')
################################################################################


# custom functions
def foo():
    pass


# handle running of the job
if __name__ == '__main__':
    # get jobfile name without extension
    _file = csutils.no_ext_basename(__file__)

    # setup logging
    log_dir = os.path.join(sources_path, '..', 'logs')
    csutils.touchdir(log_dir)
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


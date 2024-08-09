#!python3
# local_no-nb-ref_psf-import.py

# IMPORTS
# - standard imports
import os
import cProfile
import pstats
import sys

import napari

# - local imports
sources_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..'))
if sources_dir not in sys.path:
    sys.path.append(sources_dir)

import c_swain_python_utils as csutils
from utilities import *
import prairie_view_imports as pvi
import imaging_dataset as imd

# - additional imports


# LOGGER
log = csutils.get_logger(__name__)

# GLOBAL VARIABLES
# - paths
config_dir = os.path.abspath(os.path.join(sources_dir, '..', 'configs', ))

# - logging setup
debug_mode = False
default_window_level = 'info'

# - script profiling setup
do_profile = False
profile_inclusion_list = []
run_tag = '00'

# - custom global variables


# Main Script ##################################################################
def main():
    config = get_config('2p_res_measure_2.yml')
    dataset_info = config.get_dataset_info(name='20X_expander-OFF')
    pvd = pvi.load_dataset(dataset_info=dataset_info)
    dataset = imd.ImagingDataset.from_pvdataset(pvd,
                                                path=config.output_directory)

    log.info('\nDataset "{}" coordinate pitches (voxel size):', dataset.name)
    for d, coord in dataset.coordinates.items():
        log.info('> {:s} - {:.3f} {:s}', d, coord.median_pitch, coord.unit)

    log.info('Dataset "{}" dimension order: {}', dataset.name,
             dataset.dimensions)

    numpy_path = os.path.join(config.output_directory,
                              f'{dataset.name}_numpy-array.pkl')
    csutils.save_to_disk(dataset.get_image_data(), numpy_path)

    log.info('Opening dataset in napari.')
    napari_viewer = napari.Viewer(ndisplay=3)
    dataset.add_to_napari(napari_viewer)
    imd.ImagingDataset.apply_standard_napari_config(napari_viewer)
    napari.run()
    log.info('Napari session ended.')
################################################################################


# custom functions
def foo():
    pass


def get_config(config_name):
    """
    load configuation using utility class
    """
    config_path = os.path.join(config_dir, config_name)
    return ZMIAConfig(config_path)


# handle running of the job
if __name__ == '__main__':
    # get jobfile name without extension
    _file = csutils.no_ext_basename(__file__)

    # setup logging
    log_path = os.path.join(sources_dir, '..', 'logs', f'{_file}.log')
    csutils.apply_standard_logging_config(
        file_path=log_path,
        window_level='debug' if debug_mode else default_window_level,
        window_format='debug' if debug_mode else 'default')

    # run job and catch exceptions
    try:
        log.info('v' * 80)

        pr = None
        if do_profile:
            pr = cProfile.Profile()
            log.info('Initiating profiling of job with cProfile.')
            profile_path = os.path.join(
                sources_dir, '..', 'logs', f'{_file}_{run_tag}.profile')
            pr.enable()

        log.info('Beginning Job `{}`.', _file)
        main()

        if pr is not None:
            pr.disable()
            stats = pstats.Stats(pr)
            log.info('Writing profiler output to "{}".', profile_path)
            stats.dump_stats(profile_path)
            stats.sort_stats('time')
            log.info('Printing profiler output to stdout.')
            stats.print_stats(*profile_inclusion_list)
            pr = None

    except KeyboardInterrupt:
        log.warning('Job interrupted, exiting.')
    except Exception as e:
        log.critical('Job Failed.', exc_info=e)
        raise e
    else:
        log.info('Job Completed')
    finally:
        if pr is not None:
            pr.disable()
            pr.dump_stats(profile_path)
            pr.print_stats()

        log.info('^' * 80)

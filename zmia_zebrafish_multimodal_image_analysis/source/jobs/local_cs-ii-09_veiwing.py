#!python3
# local_cs-ii-04_multi-track_import_development.py

# imports
import os.path

import c_swain_python_utils as csutils
import prairie_view_imports as pvi
import imaging_dataset as imd
import napari
import numpy as np
import pprint as pp
import cProfile
import pstats


# logger
log = csutils.get_logger(__name__)

# standard global variables
sources_path = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..'))
config_dir = os.path.abspath(os.path.join(sources_path, '..', 'configs',))
debug_mode = False
default_window_level = 'debug'
do_profile = True
profile_inclusion_list = []
run_tag = '09'

# custom global variables


# Main Script ##################################################################
def main():
    config_path = os.path.join(config_dir, 'cs-ii-09_config.yml')

    pv_dataset = pvi.load_dataset(config=config_path, index=0)

    im_dataset = imd.ImagingDataset.from_pvdataset(pv_dataset)

    # viewer = napari.Viewer()
    # viewer.add_image(
    #     data=im_dataset.image_ndarray,
    #     name=im_dataset.name,
    #     colormap='magma')
    #
    # log.info('Opening a napari session.')
    # viewer.dims.axis_labels = im_dataset.dimensions
    # viewer.dims.order = tuple(range(len(im_dataset.shape)))
    # napari.run()
    # log.info('Napari session ended.')

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
                sources_path, '..', 'logs', f'{_file}_{run_tag}.profile')
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


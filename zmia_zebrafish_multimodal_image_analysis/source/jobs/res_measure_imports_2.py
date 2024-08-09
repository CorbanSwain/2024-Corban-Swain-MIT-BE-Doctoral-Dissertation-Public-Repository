#!python3
# local_cs-ii-04_multi-track_import_development.py

# imports
import os.path

import c_swain_python_utils as csutils
import prairie_view_imports as pvi
import imaging_dataset as imd
from utilities import *
import cProfile
import pstats


# logger
log = csutils.get_logger(__name__)

# standard global variables
sources_path = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..'))
config_dir = os.path.abspath(os.path.join(sources_path, '..', 'configs',))
debug_mode = False
default_window_level = 'debug'
do_profile = False
profile_inclusion_list = []
run_tag = ''

# custom global variables


# Main Script ##################################################################
def main():
    config_path = os.path.join(config_dir, '2p_res_measure_2.yml')
    set_global_config(config_path)

    all_pv_datasets = pvi.load_all_datasets(clear_cache=False,
                                            do_fail=True)

    for pvd in all_pv_datasets:
        md_reader = pvi.PVMetadataInterpreter(pvd.metadata)

        log.info(
            f'\n\ndataset info for dataset >> {pvd.name} <<\n'
            f'dX: {md_reader.x_pitch:.3f} {pvi.POSITION_UNITS}\n'
            f'dY: {md_reader.y_pitch:.3f} {pvi.POSITION_UNITS}\n'
            f'dZ: {md_reader.z_pitch:.3f} {pvi.POSITION_UNITS}\n'
            f'objective: {pvd.metadata[pvi.pvmk.OBJECTIVE_LENS]}\n'            
            f'lambda: {pvd.metadata[pvi.pvmk.LASER_WAVELENGTH]["0"]} nm\n'
            f'power: {pvd.metadata[pvi.pvmk.LASER_POWER]["0"]} pockels\n'
            f'notes: {pvd.metadata[pvi.pvmk.NOTES]}\n\n')

        im_data = imd.ImagingDataset.from_pvdataset(pvd)
        im_data.to_tiff()

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

    except KeyboardInterrupt:
        log.warning('Job interrupted, exiting.')
    except Exception as e:
        log.critical('Job Failed.', exc_info=e)
        raise e
    else:
        if pr is not None:
            pr.disable()
            stats = pstats.Stats(pr)
            log.info('Writing profiler output to "{}".', profile_path)
            stats.dump_stats(profile_path)
            stats.sort_stats('time')
            log.info('Printing profiler output to stdout.')
            stats.print_stats(*profile_inclusion_list)
            pr = None

        log.info('Job Completed')
    finally:
        if pr is not None:
            pr.disable()
            pr.dump_stats(profile_path)
            pr.print_stats()

        log.info('^' * 80)


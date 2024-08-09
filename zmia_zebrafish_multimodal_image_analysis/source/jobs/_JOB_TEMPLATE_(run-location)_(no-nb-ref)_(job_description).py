#!python3
# _JOB_TEMPLATE_(run-location)_(no-nb-ref)_(job_description).py

# IMPORTS
# - standard imports
import os
import cProfile
import pstats
import sys

# - local imports
jobs_dir = os.path.split(__file__)[0]
sources_dir = os.path.abspath(os.path.join(jobs_dir, '..'))
if sources_dir not in sys.path:
    sys.path.append(sources_dir)

from utilities import *
import video_utils
import c_swain_python_utils as csutils
import prairie_view_imports as pvi
import imaging_dataset as imd

# - additional imports


# LOGGER
log = csutils.get_logger(__name__)

# GLOBAL VARIABLES
# - paths
repo_dir = os.path.abspath(os.path.join(sources_dir, '..'))
config_dir = os.path.abspath(os.path.join(repo_dir, 'configs',))

# - logging setup
debug_mode = False
default_window_level = 'info'

# - script profiling setup
do_profile = False
profile_inclusion_list = []
run_tag = ''

# - custom global variables


# environment update
os.environ.update({'NUMEXPR_MAX_THREADS': str(os.cpu_count())})


# Main Script ##################################################################
def main():
    # small matter of programming
    pass
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
    profiler_and_path = None
    try:
        log.info('v' * 80)

        if do_profile:
            pr = cProfile.Profile()
            profile_path = os.path.join(
                sources_dir, '..', 'logs', f'{_file}_{run_tag}.profile')
            profiler_and_path = (pr, profile_path)
            log.info('Initiating profiling of job with cProfile.')
            pr.enable()

        log.info('Beginning Job `{}`.', _file)
        main()

        if profiler_and_path is not None:
            pr, profile_path = profiler_and_path
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
        if profiler_and_path is not None:
            pr, profile_path = profiler_and_path
            pr.disable()
            pr.dump_stats(profile_path)
            pr.print_stats()

        log.info('^' * 80)

#!python3
# _JOB_TEMPLATE_(run-location)_(no-nb-ref)_(job_description).py

# IMPORTS
# - standard imports
import os
import cProfile
import pstats
import sys
from time import sleep

import napari
import numpy as np

# - local imports
jobs_dir = os.path.split(__file__)[0]
sources_dir = os.path.abspath(os.path.join(jobs_dir, '..'))
if sources_dir not in sys.path:
    sys.path.append(sources_dir)

from video_utils import AVDataset, write_array_to_video_file
from utilities import *
import c_swain_python_utils as csutils

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
run_tag = '01'

# - custom global variables


# environment update
os.environ.update({'NUMEXPR_MAX_THREADS': str(os.cpu_count())})


# Main Script ##################################################################
def main():
    # load configuration file, this contains info about each dataset for loading
    config_name = 'no-nb-ref_vid-tests.yml'
    config_path = os.path.join(config_dir, config_name)
    config = ZMIAConfig(config_path)

    # make sure the output directory exists
    csutils.touchdir(config.output_directory)

    ds_info = config.get_dataset_info(name='test-vid-05')
    avi_ds = AVDataset(ds_info.full_path,
                       name=ds_info.name,
                       backend='pims')

    # log.info('Converting avi file to mp4 file...')
    # mp4_path = convert_avi_to_mp4(avi_ds.path)

    if not ask_yes_no('Do you want to skip backend tests?'):
        for backend in ['pims', 'avi-r', 'cv2', 'back?']:
            log.info('Performing tests for backend "{}"...', backend)
            try:
                avi_ds.backend = backend
            except Exception as e:
                log.info('Could not switch to backend "{}".', backend,
                         exc_info=e)
                continue

            try:
                num_frames_est = avi_ds.num_file_frames_estimate
            except Exception as e:
                log.info('Could not get num frames (estimated) for "{}".',
                         backend, exc_info=e)
            else:
                log.info('Num frames (estimated) according to "{}" = {}',
                         backend, num_frames_est)

            log.info('Attempting computation of actual num frames ...')
            try:
                num_frames = avi_ds.num_file_frames
            except Exception as e:
                log.info('Could not get actual num frames for "{}".',
                         backend, exc_info=e)
            else:
                log.info('Num frames according to "{}" = {}',
                         backend, num_frames)

    if not ask_yes_no('Do you want to skip frame clipping and save test?'):
        test_clipped_filename = f'{ds_info.name}_CLIPPED.mp4'
        test_clipped_filepath = os.path.join(config.output_directory,
                                             test_clipped_filename)

        avi_ds.backend = 'pims'
        clipped_frames, _ = avi_ds.get_frames(0, 100)
        write_array_to_video_file(video_data=clipped_frames,
                                  path=test_clipped_filepath,
                                  frame_rate_hz=avi_ds.frame_rate_hz)

        if os.path.exists(test_clipped_filepath):
            log.info('Output video file exists.')
        else:
            log.warning('Output video file does not exist.')

    if not ask_yes_no('Do you want to skip video display with each backend?'):
        for backend in ['pims', 'avi-r', 'cv2']:
            log.info('Attempting to grab all frames with "{}" backend.',
                     backend)
            avi_ds.backend = backend
            try:
                im_data, valid = avi_ds.get_frames(0,
                                                   avi_ds.num_file_frames,
                                                   allow_failed_frames=True)
            except Exception as e:
                log.info('Grabbing of frames failed with error:',
                         exc_info=e)
                continue

            log.info('Num Invalid Frames = {:d}', np.sum(np.logical_not(valid)))
            log.info('locs -> {}', np.where(np.logical_not(valid))[0])
            log.info('Displaying frames.')
            viewer = napari.Viewer()
            viewer.add_image(im_data)
            napari.run()

################################################################################


# custom functions

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


def ask_yes_no(prompt) -> bool:
    """
    returns `True` if the user's answer is YES to the prompt; any other input is
    interpreted as NO, returning `False`.
    """
    sleep(0.05)
    full_prompt = f'{prompt} (N/y)'
    response = input(full_prompt + '\n> ')

    log.debug('requested user input: "{:s}" ...', full_prompt)
    log.debug('got user response: "{:s}"', response)

    return response.lower().strip() in ('y', 'yes')


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

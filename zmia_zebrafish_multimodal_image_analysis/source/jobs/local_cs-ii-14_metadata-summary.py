#!python3
# cs-ii-14_shared-config.yml

# IMPORTS
# - standard imports
import os
from collections import namedtuple
import cProfile
import pstats
import sys

# - local imports
sources_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..'))
if sources_dir not in sys.path:
    sys.path.append(sources_dir)

from utilities import *
import c_swain_python_utils as csutils
import prairie_view_imports as pvi
import imaging_dataset as imd

# - additional imports
import functools as ft
import numpy as np
import napari
import textwrap

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
run_tag = '09'

# - custom global variables
DatasetIngestSpec = namedtuple('DatasetLoadInfo',
                               ['name',
                                'load_kwargs',
                                'napari_kwargs',
                                'metadata'])
TRANSLATE_REL_KEY = 'translate_relative_to'
TARGET_POWER_KEY = 'target_power_mw'
OLYMPUS_20X_SPOT_FWHM_um = 0.76
view_with_napari = True

# environment update
os.environ.update({'NUMEXPR_MAX_THREADS': str(os.cpu_count())})


# Main Script ##################################################################
def main():
    # load configuration file, this contains info about each dataset for loading
    config = get_config('cs-ii-14_shared-config.yml')

    # make sure the output directory exists
    csutils.touchdir(config.output_directory)

    # a list of the main datasets (built as namedtuples) for iterating through
    # in the following code
    main_dataset_list = [
        DatasetIngestSpec('bidir-tz-vid-01',
                          {},
                          {'colormap': 'bop orange',
                           TRANSLATE_REL_KEY: 'morph-ref-anterior',
                           'translate': {imd.Dim.Z: 67,
                                         imd.Dim.Y: -30,
                                         imd.Dim.X: 20}},
                          {TARGET_POWER_KEY: 50}),

        DatasetIngestSpec('tail-vid-01',
                          {'pixel_pitch_um': 11.1, 'fps': 150,
                           'reader': 'cv2', 'stride': 1},
                          {'translate': {imd.Dim.Y: -2920, imd.Dim.X: -470}},
                          {}),

        DatasetIngestSpec('morph-ref-anterior',
                          {},
                          {'colormap': 'bop blue'},
                          {TARGET_POWER_KEY: [14.5, 20.5]}),

        DatasetIngestSpec('morph-ref-anterior-ventral-append',
                          {},
                          {TRANSLATE_REL_KEY: 'morph-ref-anterior',
                           'colormap': 'bop blue'},
                          {TARGET_POWER_KEY: [14.5, 20.5]}),

        DatasetIngestSpec('morph-ref-posterior',
                          {},
                          {TRANSLATE_REL_KEY: 'morph-ref-anterior',
                           'colormap': 'bop blue'},
                          {'target_power_mw': [14.5, 20.5]}),

        DatasetIngestSpec('fixed-tissue-gcamp-stain',
                          {},
                          {},
                          {}),
    ]

    # dictionary to save ImagingDataset objects by name
    im_datasets = dict()

    # iterate through each of the datasets to load them
    for spec in main_dataset_list:
        # begin by getting the configuation's information for the dataset
        log.info('Loading dataset info for: {} > {}.',
                 config.name or '[unnamed]', spec.name)
        ds_info = config.get_dataset_info(name=spec.name)

        # determining the loading function for the dataset based on it's type
        if ds_info.type.lower() != 'two-photon':
            log.info('Skipping further processing for non-2P dataset.')
            continue

        log.info('Loading a dataset of type: {}', ds_info.type)
        ds_loader = choose_dataset_loader(ds_info.type)

        log.info('\nAbout this dataset:\n {}\n',
                 textwrap.fill(ds_info.description, 70))

        # initialize a low-level dataset object
        log.info('Loading the dataset based on the configured type.')
        ds = ds_loader(dataset_info=ds_info, **spec.load_kwargs)

        dsmd = ds.metadata
        dsmdit = pvi.PVMetadataInterpreter(dsmd)

        target_power_mw = np.array(spec.metadata[TARGET_POWER_KEY])
        objective_resolution_um = OLYMPUS_20X_SPOT_FWHM_um
        scanline_length_um = (dsmd.get_value('pixelsPerLine')
                              * dsmd.get_value(('micronsPerPixel', 'XAxis')))
        dwell_time_us = dsmd.get_value('dwellTime')
        scanline_period_us = dsmd.get_value('scanLinePeriod') * 1e6
        if 'resonant' in dsmd.get_value('activeMode').lower():
            dwell_time_us = scanline_period_us / dsmd.get_value('pixelsPerLine')
            samp_per_px_str = f'{dsmd.get_value("resonantSamplesPerPixel"):d}'
        else:
            samp_per_px_str = 'None'
        scanline_energy_density = (
            target_power_mw * scanline_period_us  # < scanline energy
            / (objective_resolution_um * scanline_length_um))  # < scanline area
        scanline_energy_density_str = np.array2string(scanline_energy_density,
                                                      precision=1)
        point_energy_density = (target_power_mw * dwell_time_us
                                / (np.pi * (objective_resolution_um/2) ** 2))
        point_energy_density_str = np.array2string(point_energy_density,
                                                   precision=1)
        point_power_density = (target_power_mw
                               / (np.pi * (objective_resolution_um/2) ** 2))
        point_power_density_str = np.array2string(point_power_density,
                                                  precision=1)
        power_str = np.array2string(target_power_mw,
                                    precision=1)



        pmt_gain_range = np.array(dsmd.get_value(('pmtGain', '1')))
        if pmt_gain_range.size > 1:
            pmt_gain_range = np.concatenate(pmt_gain_range)
            pmt_gain_range = np.array([pmt_gain_range.min(),
                                       pmt_gain_range.max()])

        pmt_gain_str = np.array2string(pmt_gain_range,
                                       precision=0)
        page_size = np.array([dsmdit.num_image_pixel_cols,
                              dsmdit.num_image_pixel_rows])
        fov_size = page_size * np.array([dsmdit.x_pitch, dsmdit.y_pitch])

        try:
            time_pitch = dsmdit.time_pitch
        except ValueError:
            time_pitch_str = 'None'
            acq_rate_str = 'None'
        else:
            time_pitch_str = f'{time_pitch:.2f}'
            if time_pitch == 0:
                acq_rate_str = 'None'
            else:
                acq_rate_str = f'{1/time_pitch:.2f}'

        try:
            num_tracks_by_z = dsmdit.get_num_tracks_by_net_z_locations()
            num_tracks_by_z = list(num_tracks_by_z.values())
            num_tracks_by_z = num_tracks_by_z[0]
        except Exception:
            num_tracks_by_z_str = 'None'
        else:
            num_tracks_by_z_str = f'{num_tracks_by_z:d}'

        twop_dataset_acq_summary = [
            ("acquisition type", '{:s}', ds.category),
            ('z acquisition mode', '{:s}',
             'bidirectional' if dsmd.get_value('bidirectionalZ')
             else 'unidirectional'),
            ('galvo scan mode', '{}', dsmd.get_value('activeMode')),
            ('objective', '{:s}', dsmd.get_value('objectiveLens')),
            ('PMT # 2 gain', '{:s}', pmt_gain_str),
            ('scanline energy density (nJ/um^2)', '{:s}',
             scanline_energy_density_str),
            ('point energy density (nJ/um^2)', '{:s}',
             point_energy_density_str),
            ('point power density (mW/um^2)', '{:s}',
             point_power_density_str),
            ('point power (mW)', '{:s}', power_str),
            ('laser wavelength (nm)', '{:d}',
             dsmd.get_value(('laserWavelength', '0'))),
            ('dwell time (us)', '{:.2f}', dwell_time_us),
            ('scanline period (us)', '{:.3f}',
             dsmd.get_value('scanLinePeriod') * 1e6),
            ('samples per pixel', '{:s}', samp_per_px_str),
            ('frame period (s)', '{:.3f}',
             dsmd.get_value('framePeriod')),
            ('image size [x y] (pixel)', '{:s}',
             np.array2string(page_size)),
            ('FOV size [x y] (um)', '{:s}', np.array2string(fov_size,
                                                            precision=2)),
            ('num z planes', '{:d}', dsmdit.z_size),
            ('x pitch (um)', '{:.2f}', dsmdit.x_pitch),
            ('y pitch (um)', '{:.2f}', dsmdit.y_pitch),
            ('z pitch (um)', '{:.1f}', dsmdit.z_pitch),
            ('z acquisition thickness (um)', '{:.1f}',
             dsmdit.z_bottom_position - dsmdit.z_top_position),
            ('num time points', '{:d}', dsmdit.time_size),
            ('time pitch (s)', '{:s}', time_pitch_str),
            ('acquisition rate (Hz)', '{:s}', acq_rate_str),
            ('num tracks by z-loc', '{:s}', num_tracks_by_z_str)
        ]

        [log.info('{:>35s} : ' + f, n, v)
         for n, f, v in twop_dataset_acq_summary]
################################################################################


# custom functions
def get_config(config_name):
    """
    load configuation using utility class
    """
    config_path = os.path.join(config_dir, config_name)
    return ZMIAConfig(config_path)


def choose_dataset_loader(dataset_type):
    """
    determine the function for loading the dataset based on the dataset type
    """
    if dataset_type == 'two-photon':
        return pvi.load_dataset
    elif dataset_type == 'behavioral-camera':
        return AVIDataset.from_dataset_info
    elif dataset_type == 'confocal':
        return NISDataset.from_dataset_info
    else:
        msg = (f'Provided dataset type does not have an associated reader, '
               f'"{dataset_type}".')
        raise ValueError(msg)


def process_multi_track(dataset_dict):
    """
    combine a dictionary of imaging dataset from different tracks by averaging
    together
    """
    im_stack = np.stack([imds.get_image_data()
                         for imds in dataset_dict.values()])
    new_im = np.mean(im_stack, axis=0, keepdims=False)
    return new_im.astype(im_stack[0].dtype)


# handle running of the job
if __name__ == '__main__':
    # get jobfile name without extension
    _file = csutils.no_ext_basename(__file__)

    # setup logging
    log_dir = os.path.join(sources_dir, '..', 'logs')
    csutils.touchdir(log_dir)
    log_path = os.path.join(log_dir, f'{_file}.log')
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

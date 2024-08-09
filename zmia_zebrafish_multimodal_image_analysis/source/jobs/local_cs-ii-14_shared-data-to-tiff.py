#!python3
# cs-ii-14_shared-config.yml

# IMPORTS
# - standard imports
import os
from collections import namedtuple
import cProfile
import pstats
import sys

from video_utils import AVIDataset

# - local imports
sources_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..'))
if sources_dir not in sys.path:
    sys.path.append(sources_dir)

from utilities import *
from video_utils import AVDataset
import c_swain_python_utils as csutils
import prairie_view_imports as pvi
import imaging_dataset as imd

# - additional imports
import functools as ft
import numpy as np
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
                               ['name', 'load_kwargs', 'napari_kwargs'])
TRANSLATE_REL_KEY = 'translate_relative_to'
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
                                         imd.Dim.X: 20}}),

        DatasetIngestSpec('tail-vid-01',
                          {'pixel_pitch_um': 11.1, 'fps': 150},
                          {'translate': {imd.Dim.Y: -2920, imd.Dim.X: -470}}),

        DatasetIngestSpec('morph-ref-anterior',
                          {},
                          {'colormap': 'bop blue'}),

        DatasetIngestSpec('morph-ref-anterior-ventral-append',
                          {},
                          {TRANSLATE_REL_KEY: 'morph-ref-anterior',
                           'colormap': 'bop blue'}),

        DatasetIngestSpec('morph-ref-posterior',
                          {},
                          {TRANSLATE_REL_KEY: 'morph-ref-anterior',
                           'colormap': 'bop blue'}),

        DatasetIngestSpec('fixed-tissue-gcamp-stain',
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
        log.info('Loading a dataset of type: {}', ds_info.type)
        ds_loader = choose_dataset_loader(ds_info.type)

        log.info('\nAbout this dataset:\n {}\n',
                 textwrap.fill(ds_info.description, 70))

        # initialize a low-level dataset object
        log.info('Loading the dataset based on the configured type.')
        ds = ds_loader(dataset_info=ds_info, **spec.load_kwargs)

        # convert the low-level dataset object into a high-level ImagingDataset
        # object
        log.info('Converting the loaded dataset into an ImagingDataset '
                 'instance.')
        im_dataset = imd.ImagingDataset.from_dataset(
            ds, path=config.output_directory)

        # if the returned value is a dictionary of datasets merge those into an
        # average. This is because of an implementation detail of how the
        # two-photon morphology datasets are acquired
        if isinstance(im_dataset, imd.ImagingDatasetDict):
            if (im_dataset.key_type
                    is imd.DatasetListKeyType.PRAIRIE_VIEW_TRACK):
                log.info('Averaging multi-track morphology reference into a '
                         'single dataset.')

                get_im_data = ft.partial(process_multi_track,
                                         im_dataset)

                im_dataset = imd.ImagingDataset(
                    path=im_dataset[0].path,
                    name=ds.name,
                    get_image_data_func=get_im_data,
                    dimensions=im_dataset[0].dimensions,
                    coordinates=im_dataset[0].coordinates,
                    position=im_dataset[0].position,
                    source_data=ds,
                    cache_after_init=True)

        # make sure everything is proceeding as expceted
        if not isinstance(im_dataset, imd.ImagingDataset):
                msg = (f'Unexpected datatype for im_dataset received, got an '
                       f'instance of {im_dataset.__class__.__name__}.')
                raise RuntimeError(msg)

        # log the annotated shape of the dataset
        log.info('\nDataset "{}" dimensionality and size:', im_dataset.name)
        for d, s in im_dataset.shape_dict.items():
            log.info('> {:s} - {:d}', d, s)

        # log the dimensionality of the voxels for the dataset
        log.info('\nDataset "{}" coordinate pitches (voxel size):', im_dataset.name)
        for d, coord in im_dataset.coordinates.items():
            if d not in (imd.Dim.X, imd.Dim.Y, imd.Dim.Z, imd.Dim.TIME):
                continue
            log.info('> {:s} - {:.3f} {:s}', d, coord.median_pitch, coord.unit)

        # save the dataset to a dict
        im_datasets[spec.name] = im_dataset
        log.info('Dataset {:s} is successfully loaded.', repr(im_dataset))

        # dave the dataset to an imagej compatible tiff file
        im_dataset.to_tiff()


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
        return AVDataset.from_dataset_info
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

#!python3
# local_no-nb-ref_analyze-pockels-speed.py

# IMPORTS
# - standard imports
import os
import cProfile
import pstats
import sys

# - local imports
import scipy.interpolate

jobs_dir = os.path.split(__file__)[0]
sources_dir = os.path.abspath(os.path.join(jobs_dir, '..'))
if sources_dir not in sys.path:
    sys.path.append(sources_dir)

from utilities import *
import c_swain_python_utils as csutils
import prairie_view_imports as pvi
import imaging_dataset as imd

# - additional imports
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import utils as snsutils
import pprint as pp
from collections import namedtuple
import enum
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools as it

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
clip_first_n_frames = 1
discard_fraction_top = 0.3
discard_fraction_bottom = 0.05
discard_fraction_lineends = 0.05
normalization_line_fraction = 0.075
norm_center_pixel_gap = 25
norm_quantile = 0.15
norm_func = np.mean
final_keep_width_pixels = 80
transition_diff_cutoff = 0.05
transition_one_ended_keep_expansion = 2
transition_high = 0.95
transition_low = 0.05

debug_plot = False

# - custom meta


class EdgeType(enum.Enum):
    RISING = 1
    FALLING = 0

    def __str__(self):
        if self is EdgeType.FALLING:
            return 'falling'
        if self is EdgeType.RISING:
            return 'rising'



MeasurementData = namedtuple(
    "MeasurementData",
    ('name', 'image_data', 'pockels', 'dwell_time_us', 'edge'))

# environment update
os.environ.update({'NUMEXPR_MAX_THREADS': str(os.cpu_count())})


# Main Script ##################################################################
def main():
    config_filename = 'pockels-speed-v1-config.yml'
    config_path = os.path.join(config_dir, config_filename)
    log.info('Loading config from "{}".', config_path)
    config = ZMIAConfig(config_path)

    csutils.touchdir(config.output_directory)

    do_clear_cache = False

    log.info('Config file has {:d} image datasets specified.',
             len(config.dataset_list))
    datasets = []
    for zdi in config.dataset_list:
        log.debug('Loading {}...', zdi.name)
        pvd = pvi.load_dataset(dataset_info=zdi, clear_cache=do_clear_cache)
        mi = pvi.PVMetadataInterpreter(pvd.metadata)
        image_dataset = imd.ImagingDataset.from_pvdataset(
            pvd,
            path=config.output_directory,
            clear_cache=do_clear_cache)
        datasets.append(MeasurementData(
            name=image_dataset.name,
            image_data=image_dataset.get_image_data(),
            pockels=round(mi.pockels),
            dwell_time_us=mi.dwell_time_us,
            edge=(EdgeType.RISING if 'rise' in zdi.name else EdgeType.FALLING)))
    log.info('All datasets successfully loaded.')

    for i, ds in enumerate(datasets):
        log.info('[{:3d}] | Dataset {:25s} - dwell time: {:5.1f} '
                 '- pockels: {:5.1f} - image shape: {} - edge: {}',
                 i,
                 ds.name,
                 ds.dwell_time_us,
                 ds.pockels,
                 ds.image_data.shape,
                 ds.edge)

    log.info('Beginning data processing')
    sns.set_theme(style='ticks', palette='deep')
    line_index_start = 0
    line_scan_dfs = []
    transition_measure_dfs = []
    for ds in datasets:
        log.info('Processing data from dataset {:s}', ds.name)
        image_data = ds.image_data.copy()
        frame_dim = 0
        y_dim = 1
        x_dim = 2

        # discard first frame
        filtered_data = image_data[clip_first_n_frames:]
        num_frames = filtered_data.shape[frame_dim]

        # trim first and last sets of scan lines
        y_size = image_data.shape[y_dim]
        line_keep_start = round(y_size * discard_fraction_top)
        line_keep_stop = y_size - round(y_size * discard_fraction_bottom)
        log.debug('Keeping lines {:d} to {:d} of {:d}',
                  line_keep_start, line_keep_stop, y_size)
        filtered_data = filtered_data[:, line_keep_start:line_keep_stop]
        log.debug('Filtered data shape after top-bottom trim: {}',
                  filtered_data.shape)

        # stack all rep frames
        new_num_frames = filtered_data.shape[frame_dim]
        new_y_size = filtered_data.shape[y_dim]
        x_size = image_data.shape[x_dim]
        filtered_data = filtered_data.reshape(
            (1, new_y_size * new_num_frames, x_size))
        filtered_data = filtered_data.squeeze()
        rep_dim = 0
        x_dim = 1
        rep_size = filtered_data.shape[rep_dim]
        log.debug('Filtered data shape after stack: {}',
                  filtered_data.shape)

        if debug_plot:
            plt.imshow(filtered_data, cmap='magma')
            plt.title(f'"{ds.name}" filtered data')
            plt.show()

        # define x coordinates
        x_raw_pixel_coord = np.arange(x_size)
        x_center = (x_size / 2) - 0.5
        x_coord = x_raw_pixel_coord - x_center
        log.debug('X Coordinates: {}', x_coord)

        # trim beginning and end of each scan line
        keep_start = round(x_size * discard_fraction_lineends)
        keep_stop = x_size - keep_start
        filtered_data = filtered_data[:, keep_start:keep_stop]
        x_coord_trim = x_coord[keep_start:keep_stop]
        log.debug('Filtered data shape after begin-end trim: {}',
                  filtered_data.shape)

        # normalize each scan line
        new_x_size = filtered_data.shape[x_dim]
        new_x_center = (new_x_size / 2) - 0.5
        num_elements_for_norm = round(new_x_size * normalization_line_fraction)
        log.debug('Converting data from {} to {}.', filtered_data.dtype, float)
        filtered_data = filtered_data.astype(float)
        keep_stop = round(new_x_center - norm_center_pixel_gap)
        keep_start = keep_stop - num_elements_for_norm
        line_start = filtered_data[:, keep_start:keep_stop].copy()
        keep_start = round(new_x_center + norm_center_pixel_gap)
        keep_stop = keep_start + num_elements_for_norm
        line_end = filtered_data[:, keep_start:keep_stop].copy()

        if ds.edge is EdgeType.FALLING:
            line_high = line_start
            line_low = line_end
        else:
            # RISING case
            line_high = line_end
            line_low = line_start

        if debug_plot:
            fig, axs = plt.subplots(1, 2)
            axs[0].set_title('Line High')
            im = axs[0].imshow(line_high, cmap='magma')
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes("right", size="10%", pad=0.5)
            plt.colorbar(im, cax)
            axs[1].set_title('Line Low')
            im = axs[1].imshow(line_low, cmap='magma')
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes("right", size="10%", pad=0.5)
            plt.colorbar(im, cax)
            plt.tight_layout()
            plt.show()

        # cutoff = np.quantile(line_high, (1 - norm_quantile),
        #                      axis=x_dim, keepdims=True)
        # high_ref_mask = line_high >= cutoff
        # high_ref_count = np.sum(high_ref_mask, axis=x_dim, keepdims=True)
        # line_high[np.logical_not(high_ref_mask)] = 0
        # high_ref = (np.sum(line_high, axis=x_dim, keepdims=True)
        #             / high_ref_count)
        #
        # cutoff = np.quantile(line_low, norm_quantile,
        #                      axis=x_dim, keepdims=True)
        # low_ref_mask = line_low <= cutoff
        # low_ref_count = np.sum(low_ref_mask, axis=x_dim, keepdims=True)
        # line_low[np.logical_not(low_ref_mask)] = 0
        # low_ref = (np.sum(line_low, axis=x_dim, keepdims=True)
        #            / low_ref_count)

        high_ref = norm_func(line_high, axis=x_dim, keepdims=True)
        low_ref = norm_func(line_low, axis=x_dim, keepdims=True)

        range_ref = high_ref - low_ref
        normed_data = (filtered_data - low_ref) / range_ref

        if debug_plot:
            plt.imshow(normed_data, cmap='magma')
            plt.title(f'"{ds.name}" normed data')
            plt.show()

            fig, axs = plt.subplots(2, 1)
            test_point = 150
            axs[0].plot(x_coord_trim, filtered_data[test_point])
            axs[0].plot(x_coord_trim[[0, -1]], np.repeat(low_ref[test_point], 2))
            axs[0].plot(x_coord_trim[[0, -1]], np.repeat(high_ref[test_point], 2))
            axs[1].plot(x_coord_trim, normed_data[test_point])
            plt.tight_layout()
            plt.show()

        log.debug('Pre-Norm data range: {} - {}',
                  filtered_data.min(), filtered_data.max())
        log.debug('Normed data range: {} - {}',
                  normed_data.min(), normed_data.max())
        log.debug('Normed data shape: {}',
                  normed_data.shape)

        # trim for speed
        keep_start = round(new_x_center - (final_keep_width_pixels / 2))
        keep_end = keep_start + final_keep_width_pixels
        normed_data_trim = normed_data[:, keep_start:keep_end]
        x_coord_trim = x_coord_trim[keep_start:keep_end]

        # get test point values
        mean_data = np.mean(normed_data_trim, axis=rep_dim)
        diff = np.abs(np.ediff1d(mean_data, to_begin=0))
        mask = diff > transition_diff_cutoff
        start_idx = get_longest_group(mask)
        group_len = np.where(np.diff(mask[start_idx:]))[0][0] + 1
        group_start = start_idx - transition_one_ended_keep_expansion
        group_end = start_idx + group_len + transition_one_ended_keep_expansion
        interp_y_data = mean_data[group_start:group_end]
        interp_x_data = x_coord_trim[group_start:group_end]
        if ds.edge is EdgeType.FALLING:
            factor = -1
        else:
            factor = 1
        test_points = [transition_high, 0.5, transition_low]
        interpolant = np.interp(factor * np.array(test_points),
                                factor * interp_y_data,
                                interp_x_data,)

        high_pos = interpolant[0]
        trans_pos = interpolant[1]
        low_pos = interpolant[2]

        if debug_plot:
            plt.plot(x_coord_trim, mean_data, '.-')
            plt.plot(x_coord_trim, diff, '.-')
            plt.plot(interp_x_data, interp_y_data, '.-')
            plt.plot([high_pos, trans_pos, low_pos], test_points, 'r.')
            plt.fill_between(x_coord_trim[group_start:group_end], 1,
                             color='g',
                             alpha=0.3)
            plt.title(ds.name)
            plt.xlim(-15, 15)
            plt.show()

        trans_delay_pixel = trans_pos
        trans_length_pixel = np.abs(high_pos - low_pos)

        # create data data frame
        normed_data_vector = normed_data_trim.ravel()
        x_data_vector = np.tile(x_coord_trim, rep_size)
        line_index_end = line_index_start + rep_size
        line_index = np.repeat(np.arange(line_index_start, line_index_end),
                               final_keep_width_pixels)
        line_index_start = line_index_end
        line_number = np.tile(
            np.repeat(np.arange(line_keep_start, line_keep_stop),
                      final_keep_width_pixels), num_frames)
        frame_index = np.repeat(np.arange(num_frames) + clip_first_n_frames,
                                 final_keep_width_pixels * new_y_size)

        line_df = pd.DataFrame({'norm_pixel_value': normed_data_vector,
                                'scan_x_position': x_data_vector,
                                'scan_y_position': line_number,
                                'frame_number': frame_index,
                                'scan_line_index': line_index})
        line_df['pockels'] = ds.pockels
        line_df['dwell_time'] = ds.dwell_time_us
        line_df['edge_type'] = ds.edge

        trans_df = pd.DataFrame([{'trans_delay_pixel': trans_delay_pixel,
                                  'trans_delay_us':
                                      trans_delay_pixel * ds.dwell_time_us,
                                  'trans_length_pixel': trans_length_pixel,
                                  'pockels': ds.pockels,
                                  'dwell_time': ds.dwell_time_us,
                                  'edge_type': ds.edge
                                 }])
        transition_measure_dfs.append(trans_df)
        line_scan_dfs.append(line_df)

    ldf = pd.concat(line_scan_dfs)
    tdf = pd.concat(transition_measure_dfs)
    rate_factor = (transition_high - transition_low) * 100
    tdf['trans_length_us'] = (tdf['trans_length_pixel']
                              * tdf['dwell_time'])
    tdf['trans_rate_per_pixel'] = (
            rate_factor / tdf['trans_length_pixel'])
    tdf['trans_rate_per_us'] = (
            rate_factor / tdf['trans_length_us'])

    tdf.to_csv(os.path.join(config.output_directory,
                            f'transition_quantification_{config.name}.csv'))
    ldf.to_csv(os.path.join(config.output_directory,
                            f'normalized_scanlines_{config.name}.csv'))

    # Initialize a grid of plots with an Axes for each walk
    tdf['dwell_time'] = pd.Categorical(tdf['dwell_time'])
    grid0 = sns.FacetGrid(
        tdf, col="edge_type", row="pockels",
        hue='trans_delay_pixel',
        palette='vlag',
        height=3,
        aspect=1.4)

    # Draw a horizontal line to show the starting point
    grid0.refline(y=0, linestyle="-", color='k')

    # Draw a line plot to show the trajectory of each random walk
    grid0.map(sns.barplot, "dwell_time", "trans_delay_pixel")
    for ax in grid0.axes.ravel():
        for c in ax.containers:
            ax.bar_label(c, fmt='%.1f', padding=2)

    # Adjust the tick positions and labels
    grid0.set_titles(template='spatial transition delay vs. dwell time'
                              '\n{col_name} edge, pockels: {row_name}')
    grid0.set_xlabels('dwell time (us)')
    grid0.set_ylabels('delay to 50% intensity (pixels)')
    grid0.set(ylim=(-14, 14))
    grid0.fig.tight_layout(w_pad=1, h_pad=2)
    label_order = list(map(snsutils.to_utf8, [grid0.hue_names[i] for i in [0, -1]]))
    grid0.add_legend(label_order=label_order,
                     # label_names=['early', 'late'],
                     title='delay direction')

    # Initialize a grid of plots with an Axes for each walk
    grid1 = sns.FacetGrid(
        tdf, col="edge_type", row="pockels",
        hue='trans_delay_us',
        palette='vlag',
        height=3,
        aspect=1.4)

    # Draw a horizontal line to show the starting point
    grid1.refline(y=0, linestyle="-", color='k')

    # Draw a line plot to show the trajectory of each random walk
    grid1.map(sns.barplot, "dwell_time", "trans_delay_us")
    for ax in grid1.axes.ravel():
        for c in ax.containers:
            ax.bar_label(c, fmt='%.1f', padding=2)

    # Adjust the tick positions and labels
    grid1.set_titles(template='temporal transition delay vs. dwell time'
                              '\n{col_name} edge, pockels: {row_name}')
    grid1.set_xlabels('dwell time (us)')
    grid1.set_ylabels('delay to 50% intensity (us)')
    grid1.set(ylim=(-15, 15),)
    grid1.fig.tight_layout(w_pad=1, h_pad=2)
    label_order = list(map(snsutils.to_utf8, [grid1.hue_names[i] for i in [0, -1]]))
    grid1.add_legend(label_order=label_order,
                     # label_names=['early', 'late'],
                     title='delay direction')

    # Initialize a grid of plots with an Axes for each walk
    grid3 = sns.FacetGrid(
        tdf, col="edge_type", row="pockels",
        hue='trans_length_pixel',
        palette='Blues',
        height=3,
        aspect=1.4)

    # Draw a horizontal line to show the starting point
    grid3.refline(y=0, linestyle="-", color='k')

    # Draw a line plot to show the trajectory of each random walk
    grid3.map(sns.barplot, "dwell_time", "trans_length_pixel")
    for ax in grid3.axes.ravel():
        for c in ax.containers:
            ax.bar_label(c, fmt='%.1f', padding=2)

    # Adjust the tick positions and labels
    grid3.set_titles(template=f'spatial {{col_name}}-time '
                              f'[{transition_low*100:.0f}% to '
                              f'{transition_high*100:.0f}%] vs. dwell time'
                              f'\npockels: {{row_name}}')
    grid3.set_xlabels('dwell time (us)')
    grid3.set_ylabels('transition length (pixel)')
    grid3.set()
    grid3.fig.tight_layout(w_pad=1, h_pad=2)

    # Initialize a grid of plots with an Axes for each walk
    grid4 = sns.FacetGrid(
        tdf, col="edge_type", row="pockels",
        hue='trans_length_us',
        palette='Blues',
        height=3,
        aspect=1.4)

    # Draw a horizontal line to show the starting point
    grid4.refline(y=0, linestyle="-", color='k')

    # Draw a line plot to show the trajectory of each random walk
    grid4.map(sns.barplot, "dwell_time", "trans_length_us")
    for ax in grid4.axes.ravel():
        for c in ax.containers:
            ax.bar_label(c, fmt='%.1f', padding=2)

    # Adjust the tick positions and labels
    grid4.set_titles(template=f'temporal {{col_name}}-time '
                              f'[{transition_low*100:.0f}% to '
                              f'{transition_high*100:.0f}%] vs. dwell time'
                              f'\npockels: {{row_name}}')
    grid4.set_xlabels('dwell time (us)')
    grid4.set_ylabels('transition time (us)')
    grid4.set()
    grid4.fig.tight_layout(w_pad=1, h_pad=2)

    # Initialize a grid of plots with an Axes for each walk
    grid2 = sns.FacetGrid(ldf, col="edge_type", row="pockels",
                          hue='dwell_time',
                          height=5,
                          aspect=1.5)

    # Draw a horizontal line to show the starting point
    grid2.refline(x=0, color='k', linestyle=":")
    grid2.refline(y=0, color='k', linestyle=":")
    grid2.refline(y=1, color='k', linestyle=":")

    # Draw a line plot to show the trajectory of each random walk
    grid2.map(sns.lineplot,
              'scan_x_position',
              'norm_pixel_value',
              errorbar='sd',
              markers='o',
              linewidth=2)

    # Adjust the tick positions and labels
    grid2.set_titles(template='scanline intensity vs position for'
                              '\n{col_name} edge, pockels: {row_name}')
    grid2.set_xlabels('scan x position (pixel)')
    grid2.set_ylabels('normalized pixel value (a.u.)')
    grid2.set(xlim=(-15, 15),
              ylim=(-0.1, 1.6),
              yticks=[0, 0.5, 1, 1.5],
              xticks=(np.arange(-7, 8) * 2))
    grid2.map(plt.grid, visible=True, color='k', alpha=0.3)
    grid2.fig.tight_layout(w_pad=1, h_pad=1)
    grid2.add_legend(title='dwell time (us)')
    plt.show()
################################################################################

# custom functions


def get_longest_group(a):
    # Get start, stop index pairs for islands/seq. of 1s
    idx_pairs = np.where(np.diff(np.hstack(([False], a, [False]))))[0].reshape(-1, 2)

    # Get the island lengths, whose argmax would give us the ID of longest island.
    # Start index of that island would be the desired output
    return idx_pairs[np.diff(idx_pairs, axis=1).argmax(), 0]


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

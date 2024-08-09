# base imports
from __future__ import annotations

import os
import sys
import argparse
import multiprocessing
import numpy as np
import psutil
import yaml
    
_file = os.path.abspath(__file__)

jobs_dir = os.path.split(_file)[0]
sources_dir = os.path.abspath(os.path.join(jobs_dir, '..'))
repo_dir = os.path.abspath(os.path.join(sources_dir, '..'))
config_dir = os.path.abspath(os.path.join(repo_dir, 'configs'))
default_data_dir = os.path.abspath(os.path.join(repo_dir, 'data'))

# add sources directory to path
if sources_dir not in sys.path:
    sys.path.append(sources_dir)

# local imports
import c_swain_python_utils as csutils
from utilities import *

# caiman imports
import cv2
try:
    cv2.setNumThreads(0)
except Exception:
    pass
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params

_basename = os.path.basename(_file)
_name = csutils.no_ext_basename(_file)

import warnings
warnings.filterwarnings("ignore") # ignoring all runtime warnings, which print for valid tifs

# logging
log = csutils.get_logger(__name__)

sources_path = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..'))
log_path = os.path.join(sources_dir, '..', 'logs', f'{_basename}.log')

debug_mode = False
default_window_level = 'debug'
csutils.apply_standard_logging_config(
    file_path=log_path,
    window_level='debug' if debug_mode else default_window_level,
    window_format='debug' if debug_mode else 'default')



def parse_arguments(args):
    filenames = []

    tiffpath = os.path.abspath(args.tiffpath)
    if os.path.isdir(tiffpath): # directory containing all slices
        names = list(os.listdir(tiffpath))
        names = [os.path.join(tiffpath, n) for n in names if n.endswith('.tif') and not n.endswith('vol.tif')] # ignoring composite tif
        filenames.extend(names)
    elif tiffpath.endswith('.tif'): # otherwise should be a file
        filenames.append(tiffpath)
    
    opts_dict = dict()

    if args.mc_params is not None:
        parampath = os.path.abspath(args.mc_params)
        assert parampath.endswith('.yml')
        for key, value in yaml.load(open(parampath), Loader=yaml.SafeLoader).items():
            opts_dict[key] = value
    else:
        opts_dict = {
            "strides": (9, 9),
            "overlaps": (9, 9),
            "max_shifts": (10, 10),
            "max_deviation_rigid": 3
        }

    opts_dict["pw_rigid"] = True
    return filenames, opts_dict

def calc_metrics(mc, filenames, idx):
    bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                    np.max(np.abs(mc.y_shifts_els)))).astype(int)
    final_size = np.subtract(mc.total_template_els.shape, 2 * bord_px_els) # remove pixels in the boundaries
    winsize = 100
    swap_dim = False
    resize_fact_flow = .2    # downsample for computing ROF

    tmpl_orig, correlations_orig, flows_orig, norms_orig, crispness_orig = cm.motion_correction.compute_metrics_motion_correction(
        filenames[idx], final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

    tmpl_els, correlations_els, flows_els, norms_els, crispness_els = cm.motion_correction.compute_metrics_motion_correction(
        mc.fname_tot_els[idx], final_size[0], final_size[1],
        swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)
    
    auc_c_before = sum([(c - max(correlations_orig)) * -1 for c in correlations_orig])
    auc_c_after = sum([(c - max(correlations_els)) * -1 for c in correlations_els])

    # just numerical metrics for now
    return auc_c_before, auc_c_after, crispness_orig, crispness_els

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tiffpath") # can be either absolute or relative
    parser.add_argument("--mc-params") # yml file
    parser.add_argument('-m', action='store_true') # if user wants metrics
    args = parser.parse_args()

    filenames, opts_dict = parse_arguments(args)

    # start cluster
    log.info("Starting cluster...")
    n_processes = np.maximum(int(psutil.cpu_count() - 1), 1)
    dview = multiprocessing.Pool(n_processes, maxtasksperchild=None)

    # create mc object and run
    mc = MotionCorrect(
        filenames, 
        dview=dview, 
        **opts_dict)
    log.info(f"Starting motion correction on {len(filenames)} slices.")
    mc.motion_correct(save_movie=True) # saves mmap file to /caiman_data/temp

    if args.m: # prints metrics
        for idx, _ in enumerate(filenames):
            log.info(f"---- metrics for slice {idx} ----")
            auc_c_before, auc_c_after, crispness_orig, crispness_els = calc_metrics(mc, filenames, idx)
            
            log.info(f'auc before: {round(auc_c_before, 2)}')
            log.info(f'auc after: {round(auc_c_after, 2)}')

            log.info('crispness before: ' + str(int(crispness_orig)))
            log.info('crispness after: ' + str(int(crispness_els)))

    log.info(f"memory map files: {mc.mmap_file}")

if __name__ == "__main__":
    main()










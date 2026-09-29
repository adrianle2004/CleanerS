"""
inference/utils.py

Shared utilities: config loading, device selection, semantic class/color
table (taken verbatim from the real NYU.py loader you shared -- this part
IS verified), and simple logging helpers.
"""

import os
import logging
import numpy as np


# NOTE: the verified class/color table now lives in visualize.py
# (DEFAULT_COLORMAP), ported directly from test_NYU.py's labeled_voxel2ply.
# Import from there instead of duplicating it here.
CLASSES = ['empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed',
           'sofa', 'table', 'tvs', 'furn', 'objs']
NUM_CLASSES = 12


def get_device(prefer_cuda=True):
    import torch                      # kept local: the viewers use this module
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def find_frame_pairs(data_dir):
    """Scan a folder of NYUxxxx_0000.bin / NYUxxxx_0000.png files and pair them
    up by their shared frame id, instead of relying on sorted-list-position
    (which silently misaligns if the two lists ever have different counts,
    a gap, or a missing file). Same logic as generate_cleaners_data.py.

    Returns a list of (frame_id, bin_path, png_path) tuples, sorted by frame_id.
    """
    bins = {}
    pngs = {}
    for fname in os.listdir(data_dir):
        frame_id = fname[:7]   # 'NYU0001_0000.bin' -> 'NYU0001'
        full = os.path.join(data_dir, fname)
        if fname.endswith('.bin'):
            bins[frame_id] = full
        elif fname.endswith('.png'):
            pngs[frame_id] = full

    common_ids = sorted(set(bins) & set(pngs))
    missing_png = sorted(set(bins) - set(pngs))
    missing_bin = sorted(set(pngs) - set(bins))
    if missing_png:
        logging.warning(f'{len(missing_png)} .bin files with no matching .png, skipped: {missing_png[:5]}...')
    if missing_bin:
        logging.warning(f'{len(missing_bin)} .png files with no matching .bin, skipped: {missing_bin[:5]}...')

    return [(fid, bins[fid], pngs[fid]) for fid in common_ids]


# ---- PLACEHOLDER: config loading ----
# The real repo uses EasyConfig / a cfgs/*.yaml file (per its pip install list:
# "pyyaml ... EasyConfig"). Once you share cfgs/*.yaml + how test_NYU.py loads
# it, replace this with the real loader.
def load_config(cfg_path):
    import yaml
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg
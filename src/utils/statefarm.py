import numpy as np
import os

from utils.utils import mkdir
from glob import glob
from shutil import copyfile
from pathlib import PurePath

np.random.seed(2017)

use_cache = 1
color_type_global = 1

DRIVER_IDS_TRAIN = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024', 'p026', 'p035', 'p039', 'p041',
                    'p042', 'p045', 'p047', 'p049', 'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                    'p075']
DRIVER_IDS_VALID = ['p081']


def get_driver_data(data_dir):
    """Get the driver data as a dictionary mapping an image name to a subject"""

    drivers = dict()
    path = os.path.join(data_dir, 'driver_imgs_list.csv')

    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        print('%d lines found in driver_imgs_list.csv' % len(lines))

        for line in lines[1:]:
            driver_id, _, img = line.split(',')
            drivers[img] = driver_id

    return drivers


def get_valid_path(train_path):
    """Convert a training path into a validation path"""

    t_path = PurePath(train_path)
    index = t_path.parts.index('train')
    parts = t_path.parts[:index] + ('valid',) + t_path.parts[index+1:]
    return str(PurePath().joinpath(*parts))


def get_sample_path(file_path):
    """Convert a file path into a sample path"""

    path = PurePath(file_path)
    index = path.parts.index('train')
    parts = path.parts[:index] + ('sample',) + path.parts[index:]
    return str(PurePath().joinpath(*parts))


def create_validation_set(data_dir):
    """Move the validation files to a separate directory"""

    train_dir = os.path.join(data_dir, 'train')
    driver_data = get_driver_data(data_dir)
    valid_set = set({img for (img, dr_id) in driver_data.items() if dr_id in DRIVER_IDS_VALID})
    train_files = glob(train_dir + '/c?/*.jpg')
    valid_files = [f for f in train_files if os.path.basename(f) in valid_set]

    for train_path in valid_files:
        valid_path = get_valid_path(train_path)
        mkdir(os.path.dirname(valid_path))
        os.rename(train_path, valid_path)


def create_sample_set(data_dir):
    """Create a sample set of the training data for quick experimentation"""

    train_dir = os.path.join(data_dir, 'train')
    driver_data = get_driver_data(data_dir)
    sample_set = set({img for (img, dr_id) in driver_data.items() if dr_id in DRIVER_IDS_TRAIN[:2]})
    valid_set = set({img for (img, dr_id) in driver_data.items() if dr_id in DRIVER_IDS_TRAIN[2:3]})

    train_files = glob(train_dir + '/c?/*.jpg')
    sample_files = [f for f in train_files if os.path.basename(f) in sample_set]
    valid_files = [f for f in train_files if os.path.basename(f) in valid_set]

    for train_path in sample_files:
        target_path = get_sample_path(train_path)
        mkdir(os.path.dirname(target_path))
        copyfile(train_path, target_path)

    for train_path in valid_files:
        target_path = get_valid_path(get_sample_path(train_path))
        mkdir(os.path.dirname(target_path))
        copyfile(train_path, target_path)

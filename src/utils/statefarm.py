# -*- coding: utf-8 -*-

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
    t_path = PurePath(train_path)
    index = t_path.parts.index('train')
    parts = t_path.parts[:index] + ('valid',) + t_path.parts[index+1:]
    return PurePath().joinpath(*parts)


def create_validation_set(data_dir):
    """Move the validation files to a separate directory"""

    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    driver_data = get_driver_data(data_dir)
    valid_set = set({img for (img, dr_id) in driver_data.items() if dr_id in DRIVER_IDS_VALID})
    train_files = glob(train_dir + '/*/*.jpg')
    valid_files = [f for f in train_files if os.path.basename(f) in valid_set]

    mkdir(valid_dir)
    for f in valid_files:
        train_path = f
        valid_path = get_valid_path(f)
        parent_dir = valid_path.parent

        mkdir(parent_dir)
        os.rename(train_path, valid_path)

# def load_train():

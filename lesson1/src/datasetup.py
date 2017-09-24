import os
import numpy as np
from glob import glob
from shutil import copyfile


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_directories(data_dir):
    mkdir(data_dir + '/valid')
    mkdir(data_dir + '/results')
    mkdir(data_dir + '/sample/train')
    mkdir(data_dir + '/sample/test')
    mkdir(data_dir + '/sample/valid')
    mkdir(data_dir + '/sample/results')
    mkdir(data_dir + '/test/unknown')


def get_rand_train_files(data_dir):
    train_dir = data_dir + '/train'
    files = glob(train_dir + '/*.jpg')
    return np.random.permutation(files)


def create_validation_set(data_dir):
    """Create a validation set from a subset of the training files"""

    files = get_rand_train_files(data_dir)

    mkdir(data_dir + '/valid')
    for i in range(2000):
        file_name = os.path.basename(files[i])
        os.rename(files[i], data_dir + '/valid/' + file_name)


def create_sample_set(data_dir):
    """Create a validation set from a subset of the training files"""

    files_train = get_rand_train_files(data_dir)
    files_valid = get_rand_train_files(data_dir)

    mkdir(data_dir + '/sample/train')
    mkdir(data_dir + '/sample/valid')

    for i in range(200):
        file_name = os.path.basename(files_train[i])
        copyfile(files_train[i], data_dir + '/sample/train/' + file_name)

    for i in range(50):
        file_name = os.path.basename(files_valid[i])
        copyfile(files_valid[i], data_dir + '/sample/valid/' + file_name)


def create_label_dirs(parent_dir):
    cats = glob(parent_dir + '/cat.*.jpg')
    dogs = glob(parent_dir + '/dog.*.jpg')

    mkdir(parent_dir + '/cats')
    mkdir(parent_dir + '/dogs')

    for cat in cats:
        os.rename(cat, parent_dir + '/cats/' + os.path.basename(cat))

    for dog in dogs:
        os.rename(dog, parent_dir + '/dogs/' + os.path.basename(dog))


def move_files_to_label_dirs(data_dir):
    """Re-arrange image files in directories according to their label"""

    create_label_dirs(data_dir + '/train')
    create_label_dirs(data_dir + '/valid')
    create_label_dirs(data_dir + '/sample/train')
    create_label_dirs(data_dir + '/sample/valid')


def create_test_set(data_dir):
    """Create a single unknown class for the test set"""

    files = glob(data_dir + '/test/*.jpg')

    mkdir(data_dir + '/test/unknown')

    for f in files:
        os.rename(f, data_dir + '/test/unknown/' + os.path.basename(f))

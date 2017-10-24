import bcolz
import numpy as np
import os

from os.path import join
from keras.preprocessing import image


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_array(fname, arr):
    carr = bcolz.carray(arr, rootdir=fname, mode='w')
    carr.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def read_img(img_id, parent_dir, target_size, img_type='jpg'):
    """Read and resize an image
    # Arguments
        img_id: the image file name without extension
        parent_dir: the parent directory
        target_size: resize the image to this size
        img_type: the image file type
    # Returns
        Image as a numpy array
    """
    path = join(parent_dir, '%s.%s' % (img_id, img_type))
    img = image.load_img(path, target_size=target_size)
    return image.img_to_array(img)


def get_steps(batch_generator):
    batch_size = batch_generator.batch_size
    samples = batch_generator.samples
    return np.ceil(samples / batch_size).astype('int')


def do_clip(arr, mx):
    """Clip (limit) the values in an array."""
    return np.clip(arr, (1 - mx) / 9, mx)

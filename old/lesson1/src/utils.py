import bcolz
import os


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_array(fname, arr):
    carr = bcolz.carray(arr, rootdir=fname, mode='w')
    carr.flush()


def load_array(fname):
    return bcolz.open(fname)[:]

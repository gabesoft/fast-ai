from __future__ import division, print_function

import vgg16
reload(vgg16)

import utils
reload(utils)

from vgg16 import Vgg16
from utils import plots

import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt

# def image_dim_ordering():
#     return 'th';

path = '../data/dogscats/'

batch_size = 64

vgg = Vgg16()
train_batches = vgg.get_batches(path + 'train', batch_size=batch_size)
valid_batches = vgg.get_batches(path + 'valid', batch_size=batch_size * 2)

vgg.finetune(train_batches)
vgg.fit(train_batches, valid_batches, batch_size, nb_epoch=1)
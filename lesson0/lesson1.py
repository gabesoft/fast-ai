from __future__ import division, print_function

import numpy as np
import utils
import vgg16

from vgg16 import Vgg16
from utils import plots

reload(vgg16)

reload(utils)

np.set_printoptions(precision=4, linewidth=100)

path = '../data/dogscats/'

batch_size = 64

vgg = Vgg16()
train_batches = vgg.get_batches(path + 'train', batch_size=batch_size)
valid_batches = vgg.get_batches(path + 'valid', batch_size=batch_size * 2)

# vgg.finetune(train_batches)
# vgg.fit(train_batches, valid_batches, batch_size, nb_epoch=1)

batches = vgg.get_batches(path + 'train', batch_size=4)
imgs, labels = next(batches)

plots(imgs, titles=labels)

vgg.predict(imgs, True)

# reload in python 3
# from importlib import reload

# import a class
# from vgg16custom import Vgg16Custom

# vgg.model.save_weights('../data/weights/vgg.20170923.113140.hdf5', by_name=True)
# vgg.model.load_weights('../data/weights/vgg.20170923.113140.hdf5', by_name=True)

# make ipython work with
# import sys
# sys.path.append('/home/gabe/.nix-profile/lib/python2.7/site-packages/')

# start ipython in vi mode
# ipython --TerminalInteractiveShell.editing_mode=vi --matplotlib


# start the shell with: nix-shell ./.conda-shell.nix
# http://www.jaakkoluttinen.fi/blog/conda-on-nixos/

# matplotlib
# %pylab

# create python env
# conda create -n env-name ipython numpy keras _cpickle bcolz pandas scipy theano

# activate
# source activate env-name

# deactivate
# source deactivate env-name

# install module
# conda install pillow

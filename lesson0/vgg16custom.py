from __future__ import division, print_function

import numpy as np
# from numpy.random import random, permutation
# from scipy import misc, ndimage
# from scipy.ndimage.interpolation import zoom

# import keras
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
# from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
# from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image

import json

from keras import backend as K
K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')

FILES_PATH = 'http://files.fast.ai/models/'
CLASS_FILE = 'imagenet_class_index.json'


def load_weights(model):
    file_path = get_file('vgg16.h5', FILES_PATH + 'vgg16.h5', cache_subdir='models')
    model.load_weights(file_path)


def vgg_preprocess(x):
    # Mean of each channel as provided by VGG researchers
    vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))

    x = x - vgg_mean     # subtract mean
    return x[:, ::-1]   # reverse axis bgr -> rgb


def get_classes():
    fname = 'imagenet_class_index.json'
    fpath = get_file(fname, FILES_PATH + fname, cache_subdir='models')
    with open(fpath) as f:
        class_dict = json.load(f)
        [class_dict[str(i)][1] for i in range(len(class_dict))]


def FCBlock(model):
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))


def ConvBlock(layers, model, filters):
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters, (3, 3), activation='relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


def get_batches(directory,
                gen=image.ImageDataGenerator(),
                shuffle=True,
                batch_size=8,
                class_mode='categorical'):

    return gen.flow_from_directory(directory,
                                   target_size=(224, 224),
                                   class_mode=class_mode,
                                   shuffle=shuffle,
                                   batch_size=batch_size)


class Vgg16Custom():
    """The VGG 16 Imagenet model"""

    def __init__(self):
        self.create_model()
        self.setup_classes()

    def setup_classes(self):
        self.classes = get_classes()

    def create_model(self):
        model = Sequential()
        model.add(Lambda(vgg_preprocess, input_shape=(3, 244, 244)))

        ConvBlock(2, model, 64)
        ConvBlock(2, model, 128)
        ConvBlock(3, model, 256)
        ConvBlock(3, model, 512)
        ConvBlock(3, model, 512)

        model.add(Flatten())
        FCBlock(model)
        FCBlock(model)
        model.add(Dense(1000, activation='softmax'))

        load_weights(model)

        self.model = model

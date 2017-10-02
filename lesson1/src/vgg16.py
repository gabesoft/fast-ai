import json
import numpy as np

from keras import optimizers
from keras.applications import VGG16
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import _obtain_input_shape

from keras import backend as K

FILE_PATH = 'http://files.fast.ai/models/'


class Vgg16():

    def __init__(self):
        self.model = VGG16()
        self.get_classes()

    def get_classes(self):
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, FILE_PATH + fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)

        self.classes = list(map(lambda x: x[1], class_dict.values()))

    def finetune(self, batches):
        model = self.model

        x = model.layers[-1].input

        model.layers.pop()
        for layer in model.layers:
            layer.trainable = False

        x = Dense(batches.num_class, activation='softmax')(x)
        img_input = model.layers[0].input

        self.model = Model(inputs=img_input, output=x)
        self.compile()
        self.classes = list(iter(batches.class_indices))

    def fit(self, train_batches, valid_batches, batch_size, nb_epoch=1):
        nb_train_steps = int(np.ceil(train_batches.samples / batch_size))
        nb_valid_steps = int(np.ceil(valid_batches.samples / batch_size))

        self.model.fit_generator(train_batches,
                                 steps_per_epoch=nb_train_steps,
                                 epochs=nb_epoch,
                                 validation_data=valid_batches,
                                 validation_steps=nb_valid_steps)

    def compile(self, lr=0.001):
        self.model.compile(optimizer=optimizers.Adam(lr=lr),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

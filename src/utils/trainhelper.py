import numpy as np
import os

from keras.models import model_from_json
from keras.preprocessing import image
from time import strftime
from utils.utils import mkdir
from keras.utils import to_categorical as onehot


def get_batches(path,
                gen=image.ImageDataGenerator(),
                shuffle=True,
                batch_size=8,
                class_mode='categorical'):

    return gen.flow_from_directory(path,
                                   target_size=(224, 224),
                                   class_mode=class_mode,
                                   shuffle=shuffle,
                                   batch_size=batch_size)


def get_classes(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    train_batches = get_batches(train_dir, shuffle=False, batch_size=1)
    valid_batches = get_batches(valid_dir, shuffle=False, batch_size=1)
    test_batches = get_batches(test_dir, shuffle=False, batch_size=1)

    train_classes = train_batches.classes
    valid_classes = valid_batches.classes

    train_labels = onehot(train_classes, train_batches.num_class)
    valid_labels = onehot(valid_classes, valid_batches.num_class)

    return (
        train_classes,
        valid_classes,
        train_labels,
        valid_labels,
        train_batches.filenames,
        valid_batches.filenames,
        test_batches.filenames
    )


def train_model(vgg, data_path, batch_size=64, epochs=3):
    t_batches = get_batches(data_path + '/train', batch_size=batch_size)
    v_batches = get_batches(data_path + '/valid', batch_size=batch_size * 2)
    results_path = data_path + '/results'

    vgg.finetune(t_batches)
    vgg.model.optimizer.lr = 0.01

    mkdir(results_path)

    t = strftime('%Y.%m.%d')
    for epoch in range(epochs):
        print("Running epoch: %d" % epoch)
        vgg.fit(t_batches, v_batches, batch_size=batch_size, nb_epoch=1)
        weights_file = 'ft%d-%s.h5' % (epoch, t)
        vgg.model.save_weights(results_path + '/' + weights_file)

    print("Completed %d fit operations" % epochs)


def test_model(vgg, data_path, batch_size=8):
    batches = get_batches(data_path,
                          shuffle=False,
                          batch_size=batch_size,
                          class_mode=None)
    steps = int(np.ceil(batches.samples/batch_size))
    preds = vgg.model.predict_generator(batches, steps)
    return batches, preds


def save_model(data_dir, model):
    json_data = model.to_json()
    cache_dir = os.path.join(data_dir, 'cache')
    t = strftime('%Y-%m-%d-%H')
    architecture_file = 'architecture-%s.json' % t
    weights_file = 'model-weights-%s.json' % t

    mkdir(cache_dir)
    open(os.path.join(cache_dir, architecture_file), 'w').write(json_data)
    model.save_weights(os.path.join(cache_dir, weights_file), overwrite=True)


def read_model(data_dir, architecture_file, weights_file):
    cache_dir = os.path.join(data_dir, 'cache')
    json_data = open(os.path.join(cache_dir, architecture_file)).read()
    model = model_from_json(json_data)
    model.load_weights(os.path.join(cache_dir, weights_file))
    return model

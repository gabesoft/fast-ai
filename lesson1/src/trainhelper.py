from keras.preprocessing import image
from time import time


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


def finetune_model(vgg, path, batch_size=64):
    batches = get_batches(path + '/train', batch_size=batch_size)
    vgg.finetune(batches)
    vgg.model.optimizer.lr = 0.01


def train_model(vgg, data_path, batch_size=64, epochs=3):
    t_batches = get_batches(data_path + '/train', batch_size=batch_size)
    v_batches = get_batches(data_path + '/valid', batch_size=batch_size * 2)
    results_path = data_path + '/results'

    for epoch in range(epochs):
        print("Running epoch: %d" % epoch)
        vgg.fit(t_batches, v_batches, batch_size=batch_size, nb_epoch=1)
        weights_file = 'ft%d-%s.h5' % (epoch, time())
        vgg.model.save_weights(results_path + '/' + weights_file)

    print("Completed %d fit operations" % epochs)
from utils.trainhelper import train_model, test_model
from utils.vgg16_wrapper import Vgg16
from utils.utils import save_array, load_array


def train_and_test(data_dir, no_of_epochs=4):
    results_dir = data_dir + '/results'
    batch_size = 64
    vgg = Vgg16()

    train_model(vgg, data_dir, batch_size, no_of_epochs)

    batches, preds = test_model(vgg, data_dir + '/test', batch_size=batch_size)

    save_array(results_dir + '/test_preds', preds)
    save_array(results_dir + '/filenames', batches.filenames)

    return batches, preds, vgg


def load_pred_data(data_dir):
    results_dir = data_dir + '/results'
    preds = load_array(results_dir + '/test_preds')
    filenames = load_array(results_dir + '/filenames')
    return filenames, preds

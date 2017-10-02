from trainhelper import train_model, test_model
from vgg16 import Vgg16
from utils import save_array, load_array

DATA_DIR = '../data'
RESULTS_DIR = DATA_DIR + '/results'


def train_and_test(no_of_epochs=4):
    batch_size = 64
    vgg = Vgg16()

    train_model(vgg, DATA_DIR, batch_size, no_of_epochs)

    batches, preds = test_model(vgg, DATA_DIR + '/test', batch_size=batch_size)

    save_array(RESULTS_DIR + '/test_preds', preds)
    save_array(RESULTS_DIR + '/filenames', batches.filenames)

    return batches, preds


def load_pred_data():
    preds = load_array(RESULTS_DIR + '/test_preds')
    filenames = load_array(RESULTS_DIR + '/filenames')
    return filenames, preds

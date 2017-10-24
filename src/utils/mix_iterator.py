import numpy as np

from utils.utils import get_steps
from keras.preprocessing.image import Iterator


class MixIterator(Iterator):
    def __init__(self, iters):
        self.iters = iters
        self.steps = sum([get_steps(x) for x in self.iters])
        self.samples = sum([x.samples for x in self.iters])

    def reset(self):
        for it in self.iters:
            it.reset()

    def __iter__(self):
        return self

    def next(self, *args, **kwargs):
        nexts = [next(it) for it in self.iters]
        batch_x = np.concatenate([n[0] for n in nexts])
        batch_y = np.concatenate([n[1] for n in nexts])
        return batch_x, batch_y

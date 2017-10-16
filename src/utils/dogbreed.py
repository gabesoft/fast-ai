import numpy as np
import os
import pandas as pd

from utils.utils import mkdir

np.random.seed(2017)

TRAIN_AMT = 0.8


def setup_train_data(data_dir):
    labels_file = os.path.join(data_dir, 'labels.csv')
    df_train = pd.read_csv(labels_file)
    df_grouped = df_train.groupby('breed').aggregate(lambda x: list(x))

    for breed in df_grouped.index:
        ids = df_grouped.id[breed]
        breed_len = len(ids)
        breed_len_train = np.ceil(breed_len * TRAIN_AMT).astype('int')

        train_ids = ids[:breed_len_train]
        valid_ids = ids[breed_len_train:]

        train_dir = os.path.join(data_dir, 'train')
        target_train_dir = os.path.join(train_dir, breed)
        target_valid_dir = os.path.join(data_dir, 'valid', breed)

        mkdir(target_train_dir)
        mkdir(target_valid_dir)

        for tid in train_ids:
            fpath = '%s.jpg' % tid
            src_path = os.path.join(train_dir, fpath)
            dst_path = os.path.join(target_train_dir, fpath)
            os.rename(src_path, dst_path)

        for vid in valid_ids:
            fpath = '%s.jpg' % vid
            src_path = os.path.join(train_dir, fpath)
            dst_path = os.path.join(target_valid_dir, fpath)
            os.rename(src_path, dst_path)

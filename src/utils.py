from __future__ import print_function
from __future__ import division

import os
import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split


def error_rate(predictions, labels):
    correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    total = predictions.shape[0]
    error = 100 - (100 * float(correct) / float(total))
    return error


def one_hot_encoding(Y, nb_labels):
    return (np.arange(nb_labels) == Y[:, None]).astype(np.float32)


def sample_batch(Xtr, Ytr, batch_size=64, nb_labels=10, dim=28, augment=False):
    num_samples = batch_size // nb_labels

    idx = []
    label = np.random.choice(nb_labels, 1)  # random starting point

    while len(idx) < batch_size:
        label_idx = np.where(Ytr == label)[0]
        idx.extend(np.random.choice(label_idx, num_samples, replace=False))
        label = (label + 1) % nb_labels

    idx = idx[:batch_size]  # truncate to batch size
    x_batch = Xtr[idx]
    y_batch = Ytr[idx]

    if augment:
        # stochastic data augmentation
        for i, img in enumerate(x_batch):

            img = img.reshape(dim, dim)  # reshape to image
            img = np.rot90(img, k=np.random.choice(4))  # randomly rotate image
            if np.random.rand() >= 0.5:
                img = np.fliplr(img)  # randomly flip image
            x_batch[i] = img.reshape((dim, dim, 1))

    return x_batch, y_batch


def get_cell_data(ground_truth_folder='./data/cell-data', dim=28):

    with open(os.path.join(ground_truth_folder, 'labels.txt')) as f:
        label_array = map(lambda x: x.split('\t'), f.read().splitlines())

    Y = np.array([int(elem[1]) for elem in label_array]) - 1

    all_files = os.listdir(ground_truth_folder)
    img_files = filter(lambda x: os.path.splitext(x)[1] == '.png', all_files)

    nb_img = len(img_files)
    nb_channels = 1

    X = np.zeros((nb_img, dim, dim, nb_channels))

    for i, img_name in enumerate(sorted(img_files)):
        img = imread(os.path.join(ground_truth_folder, img_name))    
        X[i, :, :, :] = img.reshape((dim, dim, 1))

    X = X / float(255)  # convert to floating point [0, 1]

    return train_test_split(X, Y, test_size=1000, random_state=42)

import os
import numpy as np
from six.moves.urllib.request import urlretrieve
import gzip, binascii, struct
from skimage.io import imread
from sklearn.model_selection import train_test_split


SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data/mnist-data'


def create_data_splits(
    Xtr_rows, Ytr, Xte_rows, Yte, num_training, num_validation, num_test):
    # Our validation set will be num_validation points from the original
    # training set.
    mask = range(num_training, num_training + num_validation)
    Xval_rows = Xtr_rows[mask]
    Yval = Ytr[mask]

    # Our training set will be the first num_train points from the original
    # training set.
    mask = range(num_training)
    Xtr_rows = Xtr_rows[mask]
    Ytr = Ytr[mask]

    # We use the first num_test points of the original test set as our
    # test set.
    mask = range(num_test)
    Xte_rows = Xte_rows[mask]
    Yte = Yte[mask]

    return Xtr_rows, Ytr, Xval_rows, Yval, Xte_rows, Yte


def get_all_data():
    Xtr = extract_data('train-images-idx3-ubyte.gz', 60000)
    Ytr = extract_labels('train-labels-idx1-ubyte.gz', 60000)
    Xte = extract_data('t10k-images-idx3-ubyte.gz', 10000)
    Yte = extract_labels('t10k-labels-idx1-ubyte.gz', 10000)

    num_train = Xtr.shape[0]
    num_test = Xte.shape[0]

    img_size = Xtr.shape[1]
    num_channels = 1

    # Flatten final layer
    Xtr = Xtr.reshape((num_train, img_size, img_size, num_channels))
    Xte = Xte.reshape((num_test, img_size, img_size, num_channels))

    return Xtr, Ytr, Xte, Yte


def try_download(filename):
    """
    A helper to download the data files if not present.
    """
    if not os.path.exists(WORK_DIRECTORY):
        os.mkdir(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not os.path.exists(filepath):
        filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print 'Successfully downloaded', filename , statinfo.st_size, 'bytes.'
    return filepath


def extract_data(filename, num_images, image_size=28, pixel_depth=255):
    filepath = try_download(filename)

    with gzip.open(filepath) as bytestream:
        # Skip magic number and dimensions
        bytestream.read(16)

        buf = bytestream.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (pixel_depth / 2.0)) / pixel_depth
        data = data.reshape(num_images, image_size, image_size, 1)
        return data


def extract_labels(filename, num_images, num_labels=10):
    filepath = try_download(filename)

    with gzip.open(filepath) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)

    return labels.astype(np.int8)


def get_cell_data(ground_truth_folder='./data/cell-data', IMG_SIZE=28):
    with open(os.path.join(ground_truth_folder, 'labels.txt')) as f:
        label_array = map(lambda x: x.split('\t'), f.read().splitlines())
    Y = np.array([int(elem[1]) for elem in label_array]) - 1

    all_files = os.listdir(ground_truth_folder)
    img_files = filter(lambda x: os.path.splitext(x)[1] == '.png', all_files)

    N = len(img_files)
    CHANNELS = 1

    X = np.zeros((N, IMG_SIZE, IMG_SIZE, CHANNELS))

    for i, img_name in enumerate(sorted(img_files)):
        img = imread(os.path.join(ground_truth_folder, img_name))    
        X[i, :, :, :] = img.reshape((IMG_SIZE, IMG_SIZE, 1))

    X = X / float(255)  # convert to floating point [0, 1]

    return train_test_split(X, Y, test_size=1000, random_state=42)

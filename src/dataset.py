import gzip
import numpy as np
import os
import six
from six.moves.urllib import request
import struct

from chainer import datasets


_save_dir = 'dataset'
_train_name = 'train.npz'
_test_name = 'test.npz'


def get_fashion_mnist():
    _retrieve_fashion_mnist()
    train = np.load(os.path.join(_save_dir, _train_name))
    test = np.load(os.path.join(_save_dir, _test_name))
    train_dataset = datasets.TupleDataset(train['x'].astype(np.float32), train['y'].astype(np.int32))
    test_dataset = datasets.TupleDataset(test['x'].astype(np.float32), test['y'].astype(np.int32))
    return train_dataset, test_dataset


def _make_npz(image_path, label_path, save_path):
    with gzip.open(image_path, 'rb') as f_image, gzip.open(label_path, 'rb') as f_label:
        format_id, = struct.unpack('>i', f_image.read(4))
        assert format_id == 2051
        format_id, = struct.unpack('>i', f_label.read(4))
        assert format_id == 2049

        image_num, = struct.unpack('>i', f_image.read(4))
        label_num, = struct.unpack('>i', f_label.read(4))
        assert image_num == label_num
        f_image.read(8)

        images = np.empty((image_num, 784), dtype=np.uint8)
        labels = np.empty(label_num, dtype=np.uint8)

        for i in six.moves.range(image_num):
            for j in six.moves.range(784):
                images[i, j] = ord(f_image.read(1))
            labels[i] = ord(f_label.read(1))
    np.savez_compressed(save_path, x=images, y=labels)


def _retrieve_fashion_mnist():
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    train_image = 'train-images-idx3-ubyte.gz'
    train_label = 'train-labels-idx1-ubyte.gz'
    test_image = 't10k-images-idx3-ubyte.gz'
    test_label = 't10k-labels-idx1-ubyte.gz'
    file_names = [train_image, train_label, test_image, test_label]

    if not os.path.exists(_save_dir):
        os.makedirs(_save_dir)
    for file_name in file_names:
        save_path = os.path.join(_save_dir, file_name)
        if os.path.exists(save_path):
            continue
        request.urlretrieve('{:s}{:s}'.format(base_url, file_name), save_path)

    train_path = os.path.join(_save_dir, _train_name)
    image_path = os.path.join(_save_dir, train_image)
    label_path = os.path.join(_save_dir, train_label)
    if not os.path.exists(train_path):
        _make_npz(image_path, label_path, train_path)
    test_path = os.path.join(_save_dir, _test_name)
    image_path = os.path.join(_save_dir, test_image)
    label_path = os.path.join(_save_dir, test_label)
    if not os.path.exists(test_path):
        _make_npz(image_path, label_path, test_path)

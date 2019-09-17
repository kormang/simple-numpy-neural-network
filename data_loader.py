import numpy as np
from os import path

def loadMNIST(prefix, folder):
    int_type = np.dtype('int32').newbyteorder('>')
    n_metadata_bytes = 4 * int_type.itemsize

    data = np.fromfile(path.join(folder, prefix + '-images.idx3-ubyte'), dtype = 'ubyte')
    magic_bytes, n_images, width, height = np.frombuffer( data[:n_metadata_bytes].tobytes(), int_type)
    data = data[n_metadata_bytes:].astype(dtype = 'float32').reshape([n_images, width, height])

    labels = np.fromfile(path.join(folder, prefix + '-labels.idx1-ubyte'),
                          dtype = 'ubyte')[2 * int_type.itemsize:]

    return data, labels

def to_hot_encoding(classification):
    # emulates the functionality of tf.keras.utils.to_categorical( y )
    hot_encoding = np.zeros([len(classification),
                              np.max(classification) + 1])
    hot_encoding[np.arange(len(hot_encoding)), classification] = 1
    return hot_encoding

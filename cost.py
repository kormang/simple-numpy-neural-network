import numpy as np

def softmax_cross_entropy(a, y):
    return -np.sum(np.log(np.sum(a * y, axis=1))), a - y

import numpy as np

def softmax_cross_entropy(a, y):
    return -np.log(a[np.argmax(y)]), a - y

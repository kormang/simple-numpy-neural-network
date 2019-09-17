import numpy as np
from activations import SoftmaxActivation

RANDOM_INIT_MAX = 0.01

class DenseLayer:
    def __init__(self, inputs, outputs, activation):
        self.inputs = inputs
        self.outputs = outputs
        self.activation = activation
        # self.w = None
        # self.b = None
        # self.grad_w = None
        # self.grad_b = None
        # self.a = None
        # self.v = None
        # self.a_prev = None

    def prepare_training(self):
        self.w = RANDOM_INIT_MAX * np.random.rand(self.inputs, self.outputs)
        self.b = RANDOM_INIT_MAX * np.random.rand(self.outputs)
        self.grad_w = np.zeros(self.w.shape)
        self.grad_b = np.zeros(self.b.shape)
        self.a = np.zeros(self.b.shape)
        self.v = np.zeros(self.b.shape)

    def prepare_iteration(self):
        self.grad_w[:] = 0
        self.grad_b[:] = 0

    def update(self, alpha):
        self.w -= alpha * self.grad_w
        self.b -= alpha * self.grad_b

    def forward(self, a_prev):
        self.a_prev = a_prev
        self.v[:] = a_prev @ self.w + self.b
        self.a[:] = self.activation.f(self.v)
        return self.a

    def backward(self, delta_w_next):
        da = self.activation.df(self.v, self.a)
        delta = da * delta_w_next
        self.grad_w += np.outer(self.a_prev, delta)
        self.grad_b += delta
        delta_w_prev = np.inner(self.w, delta)
        return delta_w_prev

class SoftmaxCrossEntropyLayer(DenseLayer):
    def __init__(self, inputs, outputs):
        super().__init__(inputs, outputs, SoftmaxActivation())

    def backward(self, delta_next):
        self.grad_w += np.outer(self.a_prev, delta_next)
        self.grad_b += delta_next
        delta_w_prev = np.inner(self.w, delta_next)
        return delta_w_prev

import numpy as np
from activations import SoftmaxActivation

RANDOM_INIT_MAX = 0.2

class DenseLayer:
    def __init__(self, inputs, outputs, activation):
        self.inputs = inputs
        self.outputs = outputs
        self.activation = activation

    def prepare_training(self):
        self.w = 2 * RANDOM_INIT_MAX * np.random.rand(self.inputs, self.outputs) - RANDOM_INIT_MAX
        self.b = np.zeros((self.outputs,))
        self.grad_w_delta = np.zeros(self.w.shape)
        self.grad_w = np.zeros(self.w.shape)
        self.grad_b = np.zeros(self.b.shape)
        self.a = np.zeros(self.b.shape)
        self.delta = np.zeros(self.b.shape)
        self.v = np.zeros(self.b.shape)
        self.delta_w_prev = np.zeros((self.inputs,))

    def forward(self, a_prev):
        self.a_prev = a_prev
        #self.v[:] = a_prev @ self.w + self.b
        np.matmul(a_prev, self.w, out=self.v)
        self.v += self.b
        self.activation.f(self.v, self.a)

        return self.a

    def backward(self, delta_w_next):
        self.activation.df(self.v, self.a, self.delta)
        self.delta *= delta_w_next
        np.outer(self.a_prev, self.delta, out=self.grad_w_delta)
        self.grad_w += self.grad_w_delta
        self.grad_b += self.delta
        np.matmul(self.w, delta_w_next, out=self.delta_w_prev)
        return self.delta_w_prev

class SoftmaxCrossEntropyLayer(DenseLayer):
    def __init__(self, inputs, outputs):
        super().__init__(inputs, outputs, SoftmaxActivation())

    def backward(self, delta_next):
        np.outer(self.a_prev, delta_next, out=self.grad_w_delta)
        self.grad_w += self.grad_w_delta
        self.grad_b += delta_next
        np.matmul(self.w, delta_next, out=self.delta_w_prev)
        return self.delta_w_prev

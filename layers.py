import numpy as np
from activations import SoftmaxActivation

RANDOM_INIT_MAX = 0.2

class DenseLayer:
    def __init__(self, inputs, outputs, activation):
        self.inputs = inputs
        self.outputs = outputs
        self.activation = activation

    def init_fwd_temps(self, batch_size):
        self.a = np.zeros((batch_size, self.outputs))
        self.v = np.zeros(self.a.shape)

    def init_bwd_temps(self, batch_size):
        self.delta = np.zeros(self.a.shape)
        self.grad_w_delta = np.zeros(self.w.shape)
        self.delta_w_prev = np.zeros((batch_size, self.inputs))
        self.grad_w = np.zeros(self.w.shape)
        self.grad_b = np.zeros(self.b.shape)

    def prepare_training(self, batch_size):
        self.w = 2 * RANDOM_INIT_MAX * np.random.rand(self.inputs, self.outputs) - RANDOM_INIT_MAX
        self.b = np.zeros((1, self.outputs))
        self.init_fwd_temps(batch_size)
        self.init_bwd_temps(batch_size)

    def forward(self, a_prev):
        # in case we suddenly get batch of different size we need to readjust
        if a_prev.shape[0] != self.a.shape[0]:
            self.init_fwd_temps(a_prev.shape[0])

        self.a_prev = a_prev
        #self.v[:] = a_prev @ self.w + self.b
        np.matmul(a_prev, self.w, out=self.v)
        self.v += self.b
        self.activation.f(self.v, self.a)
        return self.a

    def backward(self, delta_w_next):
        # in case we suddenly get batch of different size we need to readjust
        if delta_w_next.shape[0] != self.delta_w_prev.shape[0]:
            self.init_bwd_temps(delta_w_next.shape[0])

        self.activation.df(self.v, self.a, self.delta)
        self.delta *= delta_w_next

        # einsum magic: compute outer products of each row of a_prev with its pair in delta
        # and sum them along axis 0
        np.einsum('...ij,ik->...jk', self.a_prev, self.delta, out=self.grad_w_delta)

        self.grad_w += self.grad_w_delta
        self.grad_b += np.sum(self.delta, axis=0)

        np.matmul(delta_w_next, self.w.T, out=self.delta_w_prev)
        return self.delta_w_prev

class SoftmaxCrossEntropyLayer(DenseLayer):
    def __init__(self, inputs, outputs):
        super().__init__(inputs, outputs, SoftmaxActivation())

    def backward(self, delta_next):
        # in case we suddenly get batch of different size we need to readjust
        if delta_next.shape[0] != self.delta_w_prev.shape[0]:
            self.init_bwd_temps(delta_next.shape[0])

        # einsum magic: compute outer products of each row of a_prev with its pair in delta
        # and sum them along axis 0
        np.einsum('...ij,ik->...jk', self.a_prev, delta_next, out=self.grad_w_delta)

        self.grad_w += self.grad_w_delta
        self.grad_b += np.sum(delta_next, axis=0)

        np.matmul(delta_next, self.w.T, out=self.delta_w_prev)
        return self.delta_w_prev

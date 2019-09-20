import numpy as np

class AbstractActivation:
    def f(self, u, out_a):
        raise NotImplementedError("f not implemented")

    def df(self, u, a):
        raise NotImplementedError("f not implemented")

class ReLUActivation(AbstractActivation):
    def f(self, u, out_a):
        out_a[:] = u
        out_a[out_a < 0.0] = 0.0
        return out_a

    def df(self, u, a, out_da):
        out_da[:] = u
        out_da[out_da < 0] = 0
        out_da[out_da > 0] = 1
        return out_da


class SoftmaxActivation(AbstractActivation):
    def f(self, u, out_a):
        np.exp(u, out=out_a)
        sum_exps = np.sum(out_a, axis=1)
        np.divide(out_a, sum_exps.reshape((len(sum_exps), 1)), out=out_a)
        return out_a

    def df(self, u, a, out_da):
        out_da[:] = a * (1 - a)
        return out_da

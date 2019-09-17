import numpy as np

class AbstractActivation:
    def f(self, u):
        raise NotImplementedError("f not implemented")

    def df(self, u, a):
        raise NotImplementedError("f not implemented")

class ReLUActivation(AbstractActivation):
    def f(self, u):
        r = u.copy()
        r[r < 0.0] = 0.0
        return r

    def df(self, u, a):
        r = u.copy()
        r[r < 0] = 0
        r[r > 0] = 1
        return r


class SoftmaxActivation(AbstractActivation):
    def f(self, u):
        exps = np.exp(u)
        sum_exps = np.sum(exps)
        exps /= sum_exps
        return exps

    def df(self, u, a):
        return a * (1 - a)

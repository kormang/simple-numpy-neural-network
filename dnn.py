import numpy as np
from utils import print_progress_bar

class CrossEntropyClassifier:
    def __init__(self, layers, cost_function):
        self.layers = layers
        self.cost_function = cost_function

    def forward_pass(self, x):
        a = x

        for layer in self.layers:
            a = layer.forward(a)

        return a

    def compute_cost_forward_pass(self, x, y):
        a = self.forward_pass(x)

        # calculate cost
        return self.cost_function(a, y)

    def backward_pass(self, delta_w):
        for layer in reversed(self.layers):
            delta_w = layer.backward(delta_w)

    def single_pass(self, x, y, N):
        # forward pass
        cost, delta_w = self.compute_cost_forward_pass(x, y)

        cost /= N
        delta_w /= N

        # backward pass
        self.backward_pass(delta_w)

        return cost

    def train(self, x, y, max_iter = 100, alpha = 0.1, target_cost = 0.01):
        self.cost_history = []
        N = x.shape[0]

        for l in self.layers:
            l.prepare_training()

        for iter in range(max_iter):
            cost = 0

            for l in self.layers:
                l.prepare_iteration()

            for n in range(N):
                cost += self.single_pass(x[n], y[n], N)
                print_progress_bar(n+1, N,
                    prefix="it: {0:03}".format(iter+1),
                    suffix = "{}/{} cost: {}".format(n+1, N, cost), length = 50)

            self.cost_history.append(cost)
            if cost <= target_cost:
                break

            # this could be refactored to accomodate different optimizers
            # but it works for now, and its efficient enough
            for l in self.layers:
                l.update(alpha)

    def grad_check(self, x, y):
        for l in self.layers:
            l.prepare_training()

        cost, delta_w = self.compute_cost_forward_pass(x, y)

        # backward_pass will calculate gradients
        self.backward_pass(delta_w)

        print('Cost: ' + str(cost))
        for i, l in enumerate(self.layers):
            print('Layer ' + str(i) + ':')
            print(l.grad_b)
            print(l.grad_w)
            print('-----')

        deriv_delta = 0.00001

        for layer in self.layers:
            layer.grad_w[:] = 0
            layer.grad_b[:] = 0

            # calculate grad_w numerically:
            for i in range(layer.grad_w.shape[0]):
                for j in range(layer.grad_w.shape[1]):
                    w_orig = layer.w[i, j]

                    layer.w[i, j] = w_orig + deriv_delta
                    cost1, _ = self.compute_cost_forward_pass(x, y)
                    layer.w[i, j] = w_orig - deriv_delta
                    cost2, _ = self.compute_cost_forward_pass(x, y)

                    layer.grad_w[i, j] = (cost1 - cost2) / (2 * deriv_delta)
                    layer.w[i, j] = w_orig

            # calculate grad_b numerically:
            for i in range(layer.grad_b.shape[0]):
                b_orig = layer.b[i]

                layer.b[i] = b_orig + deriv_delta
                cost1, _ = self.compute_cost_forward_pass(x, y)
                layer.b[i] = b_orig - deriv_delta
                cost2, _ = self.compute_cost_forward_pass(x, y)

                layer.grad_b[i] = (cost1 - cost2) / (2 * deriv_delta)
                layer.b[i] = b_orig

            print('Numeric calculation:')

            for i, l in enumerate(self.layers):
                print('Layer ' + str(i) + ':')
                print(l.grad_b)
                print(l.grad_w)
                print('-----')

    def predict(self, x):
        r = np.empty((x.shape[0], self.layers[-1].outputs))
        for i in range(x.shape[0]):
            r[i, :] = self.forward_pass(x[i])
        return r

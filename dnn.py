import numpy as np
from utils import print_progress_bar

class Classifier:
    def __init__(self, layers, cost_function):
        self.layers = layers
        self.cost_function = cost_function

    def forward_pass(self, x):
        a = x

        for layer in self.layers:
            a = layer.forward(a)

        return a

    def backward_pass(self, delta_w):
        for layer in reversed(self.layers):
            delta_w = layer.backward(delta_w)

    def train(self, x, y, max_iter = 100, learning_rate = 0.1, target_acc = 0.99, batch_size = 600):
        self.cost_history = []
        N = x.shape[0]

        rnd = np.random.RandomState()

        for l in self.layers:
            l.prepare_training(batch_size)

        for iter in range(max_iter):
            seed = np.random.randint(0, 321242341)
            rnd.seed(seed)
            rnd.shuffle(x)
            rnd.seed(seed)
            rnd.shuffle(y)

            correct_guesses = 0
            for n in range(0, N, batch_size):
                yb = y[n:min(n+batch_size, N)]
                cost = 0
                for l in self.layers:
                    l.grad_w[:] = 0
                    l.grad_b[:] = 0

                # forward pass
                a = self.forward_pass(x[n:min(n+batch_size, N)])

                correct_guesses += np.sum(
                    np.argmax(a, axis=1) == np.argmax(yb, axis=1))

                # calculate cost
                sample_cost, delta_w = self.cost_function(a, yb)
                delta_w /= min(batch_size, N - n)

                # backward pass
                self.backward_pass(delta_w)

                cost += sample_cost / min(batch_size, N - n)

                # We use simple gradient descent
                # this could be refactored to accomodate different optimizers
                # but it works for now, and its efficient enough
                for l in self.layers:
                    l.w -= learning_rate * l.grad_w
                    l.b -= learning_rate * l.grad_b

                print_progress_bar(min(n+batch_size, N), N,
                    prefix="epoch: {0:03}".format(iter+1),
                    suffix = "{}/{} cost: {:.5f}".format(min(n+batch_size, N), N, cost),
                    length = 50)
                self.cost_history.append(cost)

            acc = correct_guesses / N
            print('acc: {:.2f}%'.format(100 * acc))

            if acc >= target_acc:
                break

    def grad_check(self, x, y):
        def compute_cost_forward_pass(x, y):
            a = self.forward_pass(x)
            return self.cost_function(a, y)

        for l in self.layers:
            l.prepare_training()

        cost, delta_w = compute_cost_forward_pass(x, y)

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
                    cost1, _ = compute_cost_forward_pass(x, y)
                    layer.w[i, j] = w_orig - deriv_delta
                    cost2, _ = compute_cost_forward_pass(x, y)

                    layer.grad_w[i, j] = (cost1 - cost2) / (2 * deriv_delta)
                    layer.w[i, j] = w_orig

            # calculate grad_b numerically:
            for i in range(layer.grad_b.shape[0]):
                b_orig = layer.b[i]

                layer.b[i] = b_orig + deriv_delta
                cost1, _ = compute_cost_forward_pass(x, y)
                layer.b[i] = b_orig - deriv_delta
                cost2, _ = compute_cost_forward_pass(x, y)

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
            r[i, :] = self.forward_pass(x[i:i+1])
        return r

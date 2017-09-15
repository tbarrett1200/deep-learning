from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import functions as f

func = lambda x: 5 * x + 2

train_m = 100
train_x = np.random.randn(1, train_m)
train_y = func(train_x)

test_m = 10
test_x = np.random.randn(1, test_m)
test_y = func(test_x)

class Network(object):
    def __init__(self, dims, f_output = f.linear):
        self.layers = []

        for dim in range(1,len(dims)):
            self.layers.append(Layer(dims[dim-1], dims[dim]))

        self.layers[-1].f = f_output

    def forward_propogate(self, x):
        for layer in self.layers:
            x = layer.forward(x)

    def backward_propogate(self, dA):
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def train(self, x, y, alpha, epochs, print_rate, do_print=False):
        for i in range(1, epochs+1):
            self.forward_propogate(x)
            self.backward_propogate(self.cost(x, y, d = True))

            if i%print_rate == 0 and do_print:
                print(str(i)+":",np.round(self. cost(x, y), 6))

            for l in self.layers:
                l.W = l.W - alpha * l.dW
                l.b = l.b - alpha * l.db

    def cost(self, x, y, d = False, propogate = False):

        if propogate:
            self.forward_propogate(x)

        if d:
            return self.layers[-1].a - y
        else:
            return np.squeeze(1/(2*self.layers[-1].m) * np.sum((self.layers[-1].a - y) ** 2, axis=1, keepdims=True))


class Layer(object):
    def __init__(self, previous, size, func = f.relu):
        self.W = np.random.randn(size, previous) * np.sqrt(2/previous)
        self.b = np.zeros((size, 1))

        self.x = None
        self.m = None

        self.z = None
        self.a = None

        self.dW = None
        self.db = None

        self.f = func

    def forward(self, x):
        self.x = x
        self.m = x.shape[1]
        self.z = self.W.dot(self.x) + self.b
        self.a = self.f(self.z)
        return self.a

    def backward(self, dA):
        dZ = dA * self.f(self.z, d = True)
        self.dW = 1/self.m * dZ.dot(self.x.T)
        self.db = 1/self.m * np.sum(dZ, axis=1, keepdims=True)
        return self.W.T.dot(dZ)

network = Network([1, 1])
print("Network Created")
print("---------------")
network.train(train_x, train_y, 0.1, 100, 10, do_print=True)
print("---------------")
print("train:", np.round(network.cost(train_x, train_y), 6))
print(" test:", np.round(network.cost(test_x, test_y, propogate = True), 6))
print("---------------")

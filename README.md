# Activities Log and Study Materials

## Prerequisite Knowledge

Learning Python: 5th Edition by Mark Lutz

[Python Documentation](https://docs.python.org/3/tutorial/index.html)

[Khan Academy: Linear Algebra: Vectors and Spaces](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces)
* Vectors
* Linear combinations and spans
* Linear dependence and independence
* Subspaces and the basis for a subspace
* Vector dot and cross product
* null Space and column space

Introduction to Linear Algebra: 4th Edition by Gilbert Strang (Chapters 1-3)
* Introduction to Vectors
* Solving Linear Equations
* Vector Spaces and Subspaces

## Neural Networks

[Coursera: Neural Networks and Deep Learning](https://www.coursera.org/learn/convolutional-neural-networks/home/welcome)
(4 weeks, 3 to 6 hours/week)
* Introduction to deep learning
* Neural Networks Basics
* Shallow neural networks
* Deep Neural Networks

[Coursera: Improving Deep Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks/home/welcome)
(3 weeks, 3 to 6 hours/week)
* Practical aspects of Deep Learning
* Optimization algorithms
* Hyperparameter tuning, Batch Normalization and Programming Frameworks

[Coursera: Structuring Machine Learning Projects](https://www.coursera.org/learn/convolutional-neural-networks/home/welcome)
(2 weeks, 3 to 4 hours/week)
* ML Strategy (1)
* ML Strategy (2)

[Stanford CS 231 Online Notes: Convolutional Neural Networks (Module 1)](http://cs231n.github.io)
* Image Classification: Data-driven Approach, k-Nearest Neighbor, train/val/test splits
* Linear classification: Support Vector Machine, Softmax
* Optimization: Stochastic Gradient Descent
* Backpropagation, Intuitions
* Neural Networks Part 1: Setting up the Architecture
* Neural Networks Part 2: Setting up the Data and the Loss
* Neural Networks Part 3: Learning and Evaluation

Deep Learning Book by Ian Goodfellow
* Applied Math and Machine Learning Basics
* Linear Algebra
* Probability and Information Theory
* Machine Learning Basics

## Frameworks

[Tensorflow Documentation](https://www.tensorflow.org/get_started/)
* Getting Started with Tensorflow (Iris Data Set)
* MNIST for ML Beginners
* Deep MNIST for Experts
* Tensorflow Mechanics 101

# Projects

## Homespun Deep Artifical Neural Network Framework
Use application of knowledge of Neural Networks to implement a system to perform multi-variable function regression on sample data points. Currently in progress is extending the system to perform classification as well as capabilities for optimization algorithms.

``` Python
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
```
Code Excerpt for Basic Implementation of a Neural Netork Layer

``` Python
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
```
Code Excerpt for Basic Implementation of a complete Neural Netork

# Moving Forward
## Learn about more advanced neural network architectures
* Learn about Convolutional Networks ande Recurrent Networks in more detail
* Complete Coursera course on Convolutional Neural Networks
* Read Module 2 of Stanford notes on convolution
* Read Chapters in Deep Learning Book on Convolution
* Implement convolutional layers in Python with NumPy
## Complete Independent Project
* Use Neural Network to create an intelligent, learning CPU opponent in game.

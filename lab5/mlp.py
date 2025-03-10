import numpy as np

def relu(z):
    return np.maximum(z, 0)

def relu_derivative(z):
    return 1 * (z > 0)


class MultilayerPerceptron:

    def __init__(self, layers_sizes, regularization):
        self.layers_sizes = layers_sizes
        self.layers = len(layers_sizes)
        self.reg_lambda = regularization

        self.initialise_weights()

    def initialise_weights(self):
        self.theta_weights = []

        for layer in range(self.layers - 1):
            current_layer_size = self.layers_sizes[layer]
            next_layer_size = self.layers_sizes[layer + 1]

            standard = np.sqrt(2.0 / current_layer_size)
            thetas = np.random.randn(next_layer_size, current_layer_size + 1) * standard

            self.theta_weights.append(thetas)

        return self.theta_weights

    def backpropagation(self, X, Y):
        activated_dz = lambda x: relu_derivative(x)

        n_samples = X.shape[0]
        A, Z = self.feedforward(X)

        deltas = [None] * self.layers
        deltas[-1] = A[-1] - Y

        for layer in range(self.layers - 2, 0, -1):
            theta = self.theta_weights[layer][:, 1:]
            deltas[layer] = (deltas[layer + 1] @ theta) * activated_dz(Z[layer])

        gradients = []
        for layer in range(self.layers - 1):
            input_layer = A[layer]
            gradient = deltas[layer + 1].T @ input_layer / n_samples
            gradient[:, 1:] += (self.reg_lambda * self.theta_weights[layer][:, 1:]) / n_samples
            gradients.append(gradient)

        return gradients

    def feedforward(self, X):
        A = [None] * self.layers
        Z = [None] * self.layers
        input_layer = X

        for layer in range(self.layers - 1):
            n_samples = input_layer.shape[0]
            input_layer = np.concatenate((np.ones([n_samples, 1]), input_layer), axis=1)
            A[layer] = input_layer
            Z[layer + 1] = np.matmul(input_layer, self.theta_weights[layer].transpose())

            output_layer = relu(Z[layer + 1])

            input_layer = output_layer

        A[-1] = output_layer  # linear activation

        return A, Z

    def train(self, X, Y, iterations, learning_rate=0.01):
        for iteration in range(iterations):
            gradients = self.backpropagation(X, Y)
            for i in range(len(self.theta_weights)):
                self.theta_weights[i] -= learning_rate * gradients[i]

    def predict(self, X):
        A, Z = self.feedforward(X)
        Y_pred = A[-1]
        return Y_pred

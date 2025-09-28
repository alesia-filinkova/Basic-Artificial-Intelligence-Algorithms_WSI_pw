import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

class Perceptron:
    def __init__(self, input_size, activation="relu"):
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0
        self.activation_name = activation
        
        if activation == "relu":
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative == sigmoid_derivative
        elif activation == "linear":
            self.activation = linear
            self.activation_derivative = linear_derivative
        else:
            raise ValueError("Activation must be 'relu' or 'sigmoid'")
        
        self.input = None
        self.z = None
        self.output = None
        
    def forward(self, x):
        self.input = x
        self.z = np.dot(self.weights, x) + self.bias
        self.output = self.activation(self.z)
        return self.output
    
    def backward(self, grad_output, learning_rate):
        grad_activation = self.activation_derivative(self.output)
        grad_z = grad_output * grad_activation
        
        grad_weights = grad_z * self.input
        grad_bias = grad_z
        grad_input = grad_z * self.weights
        
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        return grad_input

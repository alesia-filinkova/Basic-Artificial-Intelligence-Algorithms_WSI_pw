import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, iteration_count=1000):
        self.learning_rate = learning_rate
        self.iteration_count = iteration_count
        self.weights = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient_descent(self, X, y):
        m = len(y)
        for _ in range(self.iteration_count):
            predictions = self.predict_proba(X)
            measure_error = predictions - y
            gradient = (1 / m) * np.dot(X.T, measure_error)
            self.weights -= self.learning_rate * gradient
        return self.weights

    def fit(self, X, y):
        features = len(X[1])
        self.weights = np.zeros((features, 1))
        self.gradient_descent(X, y)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights)
        return self.sigmoid(linear_model)
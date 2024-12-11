import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, iteration_count=1000):
        self.learning_rate = learning_rate
        self.iteration_count = iteration_count
        self.weights = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def fit(self, X, y):
        pass

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights)
        return self.sigmoid(linear_model)
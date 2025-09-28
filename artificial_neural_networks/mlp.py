from perceptron import Perceptron
import numpy as np

class MultilayerPerceptron:

    def __init__(self, layer_sizes, activations, learning_rate=0.01):
        """
        layer_sizes: [11, 32, 16, 1]
        activations: ["relu", "relu", "linear"]
        """
        self.layers = []
        self.learning_rate = learning_rate
        
        for i in range(1, len(layer_sizes)):
            layer = [Perceptron(layer_sizes[i-1], activation=activations[i-1])
                     for _ in range(layer_sizes[i])]
            self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = np.array([neuron.forward(x) for neuron in layer])
        return x
    
    def backward(self, y_true, y_pred):
        # gradient for MSE
        grad = y_pred - y_true
        for layer in reversed(self.layers):
            new_grad = np.zeros(len(layer[0].input))
            for i, neuron in enumerate(layer):
                new_grad += neuron.backward(grad[i], self.learning_rate)
            grad = new_grad
    
    def train(self, X, y, X_val=None, y_val=None, epochs=100):
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(epochs):
            total_loss = 0
            for xi, yi in zip(X, y):
                y_pred = self.forward(xi)
                loss = np.mean((yi - y_pred) ** 2)
                total_loss += loss
                self.backward(yi, y_pred)
                
            avg_train_loss = total_loss / len(X)
            history["train_loss"].append(avg_train_loss)
            
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_loss = np.mean((y_val - y_val_pred) ** 2)
                history["val_loss"].append(val_loss)
            
            if epoch % 10 == 0:
                if X_val is not None:
                    print(f"Epoch {epoch}, train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")
                else:
                    print(f"Epoch {epoch}, train_loss={avg_train_loss:.4f}")

        return history
        
    def predict(self, X):
        predictions = []
        for xi in X:
            y_pred = self.forward(xi)
            if len(y_pred) == 1:
                y_pred = y_pred[0]
            predictions.append(y_pred)
        return np.array(predictions)
        
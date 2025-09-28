from mlp import MultilayerPerceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
y = wine_quality.data.targets

X = X.to_numpy()
y = y.to_numpy().reshape(-1, 1)

# data normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
layers_sizes = [X_train.shape[1], 9, 3, 1]
regularization = 0.05
iterations = 5000
learning_rate = 0.1

# mlp = MultilayerPerceptron(layers_sizes, regularization)
# mlp.train(X_train, y_train, iterations, learning_rate)
mlp = MultilayerPerceptron(
    layer_sizes=[11, 32, 16, 1],
    activations=["sigmoid", "sigmoid", "linear"],  # линейный выход
    learning_rate=0.01
)

history = mlp.train(X_train, y_train, X_val=X_test, y_val=y_test, epochs=200)

plt.figure(figsize=(8, 6))
plt.plot(history["train_loss"], label="Train MSE", color="blue")
if history["val_loss"]:
    plt.plot(history["val_loss"], label="Validation MSE", color="orange")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

y_pred = mlp.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'R² for relu activation: {r2}')
    
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Prediction')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title(f'Predictions vs True Values (relu activation)')
plt.legend()
plt.show()

errors = y_test - y_pred
print(f'Mean Absolute Error for relu activation: {np.mean(np.abs(errors))}')
print(f'Mean Squared Error for relu activation: {np.mean(errors**2)}')

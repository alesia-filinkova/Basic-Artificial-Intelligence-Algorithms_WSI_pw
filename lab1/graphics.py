import matplotlib.pyplot as plt
import sympy as sp
import numpy as np


def graphic_f(upper_bound, func, extremum, history, A, B):
    step = sp.Float(0.1)
    x_vals = []
    current_x = -upper_bound
    while current_x <= upper_bound:
        x_vals.append(current_x)
        current_x += step

    y_vals = [func(x, A, B) for x in x_vals]

    plt.figure(figsize=(10, 6))
    plt.plot([float(x) for x in x_vals], [float(y) for y in y_vals], label=f'f(x) = {A}x + {B}sin(x)')

    # for i in range(0, len(history) - 1, 100):
    #     start_x = float(history[i])
    #     end_x = float(history[i + 1])
    #     start_y = float(func(start_x, A, B))
    #     end_y = float(func(end_x, A, B))

    #     plt.arrow(start_x, start_y,
    #               end_x - start_x,
    #               end_y - start_y,
    #               head_width=0.5, head_length=3, fc='k', ec='k')

    plt.scatter(history[0], func(sp.Float(history[0]), A, B), color='red', label="Start Point")
    plt.scatter(extremum, func(sp.Float(extremum), A, B), color='green', label="Extremum")

    plt.title("Gradient Descent on f(x)")
    plt.legend()
    plt.show()


def graphic_g(upper_bound, func, extremum, history, C):
    x_arr = [float(i) for i in np.arange(-upper_bound, upper_bound, 0.1)]
    y_arr = [float(i) for i in np.arange(-upper_bound, upper_bound, 0.1)]

    X, Y = np.meshgrid(x_arr, y_arr)
    Z = np.empty(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = float(func(float(X[i, j]), float(Y[i, j]), C))

    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z, levels=100, cmap='viridis')
    plt.colorbar()

    for i in range(0, len(history) - 1, 25):
        plt.arrow(history[i][0][0], history[i][0][1],
                  history[i + 1][0][0] - history[i][0][0],
                  history[i + 1][0][1] - history[i][0][1],
                  head_width=0.05, head_length=0.05, fc='k', ec='k')
        print(history[i][0][0])
        print(history[i][0][1])
        print(history[i+1][0][0])
        print(history[i+1][0][1])

    plt.scatter(history[0][0][0], history[0][0][1], color='yellow', label='Start Point')
    plt.scatter(extremum[0][0], extremum[0][1], color='red', label='Extremum')

    plt.title('Function g(x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.tight_layout()
    plt.show()

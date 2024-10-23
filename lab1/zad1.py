import sys
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def f(x, A, B):
    return A * x + B * sp.sin(x)


def g(x, y, C):
    return (C * x * y) / (sp.exp(x**2) + y**2)


def function_f(A, B):
    upper_bound = 4 * np.pi
    learning_rate_f = 0.001
    max_steps_f = 10000
    start_point_f = np.random.uniform(-upper_bound, upper_bound)
    extremum_f, history_f = grad_descent(f, gradient,
                                         [start_point_f], learning_rate_f,
                                         max_steps_f, upper_bound, A, B)
    print(extremum_f, history_f[0])
    step = sp.Float(0.1)
    x_vals = []
    current_x = -upper_bound
    while current_x <= upper_bound:
        x_vals.append(current_x)
        current_x += step

    y_vals = [f(x, A, B) for x in x_vals]

    plt.figure(figsize=(10, 6))
    plt.plot([float(x) for x in x_vals], [float(y) for y in y_vals], label=f'f(x) = {A}x + {B}sin(x)')
    plt.scatter(history_f[0], f(sp.Float(history_f[0]), A, B), color='red', label="Start Point")
    plt.scatter(extremum_f, f(sp.Float(extremum_f), A, B), color='green', label="Extremum")

    for i in range(0, len(history_f) - 1, 1000):
        start_x = float(history_f[i])
        end_x = float(history_f[i + 1])
        start_y = float(f(start_x, A, B))
        end_y = float(f(end_x, A, B))

        plt.arrow(start_x, start_y,
                  end_x - start_x,
                  end_y - start_y,
                  head_width=0.5, head_length=1, fc='k', ec='k')

    plt.title("Gradient Descent on f(x)")
    plt.legend()
    plt.show()


def function_g(C):
    upper_bound = 2
    learning_rate_g = 0.001
    max_steps_g = 100
    start_point_g = np.random.uniform(-upper_bound, upper_bound, size=2)
    extremum_g, history_g = grad_descent(g, gradient,
                                         [start_point_g],
                                         learning_rate_g, max_steps_g,
                                         upper_bound, C)
    print(extremum_g, history_g[0])
    
    # x_vals = np.linspace(-upper_bound, upper_bound, 100)
    # y_vals = np.linspace(-upper_bound, upper_bound, 100)
    # X, Y = np.meshgrid(x_vals, y_vals)
    # Z = g(X, Y, C)
    
    # plt.figure(figsize=(10, 6))
    # plt.contour(X, Y, Z, levels=50, cmap='viridis')
    # plt.scatter(history_g[0][0], history_g[0][1], color='red', label="Start Point")
    # plt.scatter(extremum_g[0], extremum_g[1], color='green', label="Extremum")
    
    # # Добавление стрелок для направления градиента каждые 100 шагов
    # for i in range(0, len(history_g), 100):
    #     dx = history_g[i+1][0] - history_g[i][0]
    #     dy = history_g[i+1][1] - history_g[i][1]
    #     plt.arrow(history_g[i][0], history_g[i][1], dx, dy, head_width=0.05, head_length=0.05)

    # plt.title("Gradient Descent on g(x, y)")
    # plt.legend()
    # plt.show()


def gradient(func, symbols, *args):
    if len(args) == 3:
        grad_fct = sp.diff(f(symbols[0], args[1], args[2]), symbols[0])
        grad = grad_fct.subs(symbols[0], args[0])
        return float(grad)
    else:
        grad_fct_x = sp.diff(g(symbols[0], args[0][1], args[-1]), symbols[0])
        grad_fct_y = sp.diff(g(symbols[1], args[0][0], args[-1]), symbols[1])
        grad_x = float(grad_fct_x.subs(symbols[0], args[0][0]))
        grad_y = float(grad_fct_y.subs(symbols[1], args[0][1]))
        return np.array([grad_x, grad_y])


def grad_descent(func, grad_func, start_point, learning_rate, max_steps,
                 upper_bound, *args):
    x_symb, y_symb = sp.symbols('x y')
    symbols = [x_symb, y_symb]
    point = np.array(start_point)
    history = [point.copy()]
    count_it = 0
    grad = grad_func(func, symbols, *point, *args)
    while count_it < max_steps and np.linalg.norm(grad) > 1e-6:
        grad = grad_func(func, symbols, *point, *args)
        point = point - learning_rate * grad
        point = np.clip(point, -upper_bound, upper_bound)
        history.append(point.copy())
        count_it += 1

    return point, history


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 zad1.py index")
        sys.exit(1)

    index = sys.argv[1]
    A, B, C = index[-3:][::-1]
    function_f(int(A), int(B))
    function_g(int(C))


if __name__ == "__main__":
    main()

import sys
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def f(x, A, B):
    return A * x + B * sp.sin(x)


def g(x, y, C):
    return (C * x * y) / (np.exp(x**2) + y**2)


def function_f(A, B):
    UPPER_BOUND = 4 * np.pi
    learning_rate_f = 0.001
    max_steps_f = 10000
    start_point_f = np.random.uniform(-UPPER_BOUND, UPPER_BOUND)
    extremum_f, history_f = grad_descent(f, gradient,
                                         [start_point_f], learning_rate_f,
                                         max_steps_f, UPPER_BOUND, A, B)
    print(extremum_f, history_f[0])


def function_g(C):
    UPPER_BOUND = 2
    learning_rate_g = 0.001
    max_steps_g = 10000
    start_point_g = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=2)
    extremum_g, history_g = grad_descent(g, gradient,
                                         [start_point_g],
                                         learning_rate_g, max_steps_g,
                                         UPPER_BOUND, C)
    print(extremum_g, history_g[0])


def gradient(func, symbols, *args):
    if type(args[1]) is int:
        grad_fct = sp.diff(f(symbols[0], args[1], args[2]), symbols[0])
        grad = grad_fct.subs(symbols[0], args[0])
        # print(grad)
        return float(grad)
    elif type(args[1]) is not int:
        grad_fct_x, grad_fct_y = sp.diff(g(symbols[0], args[-1]), symbols[0])
        grad_x, grad_y = grad_fct.subs(symbols[0], args[0])
        return [grad_x, grad_y]
    else:
        raise ValueError("The numbers of values can be only 1 or 2")


def grad_descent(func, grad_func, start_point, learning_rate, max_steps,
                 UPPER_BOUND, *args):
    x_symb, y_symb = sp.symbols('x y')
    symbols = [x_symb, y_symb]
    point = np.array(start_point)
    history = [point.copy()]
    count_it = 0
    print(*point)
    grad = grad_func(func, symbols, *point, *args)
    while count_it < max_steps and np.linalg.norm(grad) > 1e-6:
        # print(*point)
        grad = grad_func(func, symbols, *point, *args)
        point = point - learning_rate * grad
        point = np.clip(point, -UPPER_BOUND, UPPER_BOUND)
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

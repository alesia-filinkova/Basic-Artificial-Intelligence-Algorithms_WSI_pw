import sys
import numpy as np
import sympy as sp
import graphics


def f(x, A, B):
    return A * x + B * sp.sin(x)


def g(x, y, C):
    return (C * x * y) / (sp.exp(x**2 + y**2))


def function_f(A, B, learning_rate_f, max_steps_f):
    upper_bound = 4 * np.pi
    start_point_f = np.random.uniform(-upper_bound, upper_bound)
    extremum_f, history_f = grad_descent(f, gradient,
                                         [start_point_f], learning_rate_f,
                                         max_steps_f, upper_bound, A, B)
    print(extremum_f, history_f[0])
    graphics.graphic_f(upper_bound, f, extremum_f, history_f, A, B)


def function_g(C, learning_rate_g, max_steps_g):
    upper_bound = 2
    start_point_g = np.random.uniform(-upper_bound, upper_bound, size=2)
    extremum_g, history_g = grad_descent(g, gradient,
                                         [start_point_g],
                                         learning_rate_g, max_steps_g,
                                         upper_bound, C)
    print(extremum_g, history_g[0])
    graphics.graphic_g(upper_bound, g, extremum_g, history_g, C)


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
    if len(sys.argv) != 6:
        print("Usage: python3 zad1.py index, learning_rate_f, max_steps_f, learning_rate_g, max_steps_g")
        sys.exit(1)

    index = sys.argv[1]
    A, B, C = index[-3:][::-1]
    function_f(int(A), int(B), float(sys.argv[2]), int(sys.argv[3]))
    function_g(int(C), float(sys.argv[4]), int(sys.argv[5]))


if __name__ == "__main__":
    main()

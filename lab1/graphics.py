import matplotlib.pyplot as plt
import sympy as sp


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
    plt.scatter(history[0], func(sp.Float(history[0]), A, B), color='red', label="Start Point")
    plt.scatter(extremum, func(sp.Float(extremum), A, B), color='green', label="Extremum")

    for i in range(0, len(history) - 1, 1000):
        start_x = float(history[i])
        end_x = float(history[i + 1])
        start_y = float(func(start_x, A, B))
        end_y = float(func(end_x, A, B))

        plt.arrow(start_x, start_y,
                  end_x - start_x,
                  end_y - start_y,
                  head_width=0.5, head_length=1, fc='k', ec='k')

    plt.title("Gradient Descent on f(x)")
    plt.legend()
    plt.show()

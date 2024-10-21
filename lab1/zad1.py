import sys
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1e-6


def function_f(A, B):
    UPPER_LIMIT = np.pi * 4
    x = np.random.uniform(-UPPER_LIMIT, UPPER_LIMIT)
    print(f(x, A, B), x, A, B)
    print(derivate_f(x, A, B), x, A, B)


def f(x, A, B):
    return A * x + B * np.sin(x)


def gradient_f(x, A, B):
    # df_dx = A + B * np.cos(x)
    
    return df_dx


def function_g(x, y, C):
    return (C * x * y) / (np.exp(x**2) + y**2)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 zad1.py index")
        sys.exit(1)

    index = sys.argv[1]
    A, B, C = index[-3:][::-1]
    function_f(int(A), int(B))


if __name__ == "__main__":
    main()

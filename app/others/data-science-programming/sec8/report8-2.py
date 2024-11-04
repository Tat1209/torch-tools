import numpy as np

x_range = (0, 1)
y_0 = 1

h = 0.1

def dy_dx(x, y):
    return y

def rk4_method(x_range, y_0, h):
    y_n = []
    for i, x in enumerate(np.arange(x_range[0], x_range[1] + h, h)):
        if i == 0:
            y = y_0
        else:
            k1 = dy_dx(x, y)
            k2 = dy_dx(x + h / 2, y + h * k1 / 2)
            k3 = dy_dx(x + h / 2, y + h * k2 / 2)
            k4 = dy_dx(x + h, y + h * k3)
            y += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        y_n.append(y)
        print(f"{x=:.4f}, {y=:.4f}")
    return y_n
        

y_n = rk4_method(x_range, y_0, h)
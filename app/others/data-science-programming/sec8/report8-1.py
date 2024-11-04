import matplotlib.pyplot as plt
import numpy as np

x_range = (0, 1)
y_0 = 1

h = 0.1

def dy_dx(x, y):
    return y

def euler_method(x_range, y_0, h):
    x_n = np.arange(x_range[0], x_range[1] + h, h)
    y_n = []
    for i, x in enumerate(x_n):
        if i == 0:
            y = y_0
        else:
            y += dy_dx(x, y) * h
        y_n.append(y)
        print(f"{x=:.4f}, {y=:.4f}")
    return x_n, y_n

x_n, y_n = euler_method(x_range, y_0, h)

fig, ax = plt.subplots()
ax.plot(x_n, y_n)

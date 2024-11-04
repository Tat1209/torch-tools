import numpy as  np

N = 46
A = np.array([
[10*N, 2, 3, 4, 5, 6],
[2, 30*N, 4, 5, 6, 7],
[3, 4, 50*N, 6, 7, 8],
[4, 5, 6, 70*N, 8, 9],
[5, 6, 7, 8, 90*N, 10],
[6, 7, 8, 9, 10, 110*N]
])
b = np.array([1, 4, 2, 8, 5, 7])
x = np.linalg.solve(A, b)

print(x)
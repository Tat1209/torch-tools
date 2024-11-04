
N = 46
A = [
[10*N, 2, 3, 4, 5, 6],
[2, 30*N, 4, 5, 6, 7],
[3, 4, 50*N, 6, 7, 8],
[4, 5, 6, 70*N, 8, 9],
[5, 6, 7, 8, 90*N, 10],
[6, 7, 8, 9, 10, 110*N]
]
b = [1, 4, 2, 8, 5, 7]

def jacobi(A, b, iters=1000):
    n = len(A)
    x = [0] * n
    x_new = x[:]
    
    for _ in range(iters):
        for i in range(n):
            sigma = 0
            for j in range(n):
                if i != j:
                    sigma += A[i][j] * x[j]
            x_new[i] = (b[i] - sigma) / A[i][i]
        
        x = x_new[:]
    
    return x


print(jacobi(A, b, iters=1000))

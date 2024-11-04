# -*- coding: utf-8 -*-
"""
laplace.py program
Laplace equation solving program 
Solve Laplace equation by iterative method
Usage c: \> python laplace.py
"""
# Import module
import math

# Constant
LIMIT = 1000 # number of iterations
N = 101 # Number of divisions in the x-axis direction
M = 101 # Number of divisions in the y-axis direction

# Define subcontract function
# iteration () function
def iteration(u):
    "" "One iteration" ""
    u_next = [[0 for i in range (N)] for j in range (M)] # next step uij
    # Calculate next step value
    for i in range(1, N - 1):
        for j in range(1, M - 1):
            u_next[i][j] = (u[i][j - 1] + u[i -1][j] + u[i + 1][j]
                            + u[i][j + 1]) / 4

    # update uij
    for i in range(1, N - 1):
        for j in range(1, M - 1):
            u[i][j] = u_next[i][j]
# end of iteration () function

# Main execution part
u = [[0 for i in range (N)] for j in range (M)] # uij initialization
for i in range(M):
    u[0][i] = math.sin(2 * math.pi * i / (M - 1))

# Iterative calculation
for i in range(LIMIT):
    iteration(u)

print (u) # print result
# end of laplace.py
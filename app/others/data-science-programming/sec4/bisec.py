# -*- coding: utf-8 -*-
"""
bisec.py program
Solving equations by bisection method
Usage c: \> python bisec.py
"""
# Global variable
a = 2          # f(x)=x*x-a
LIMIT = 1e-20  # Termination condition

# Functions
# f()
def f(x):
    """Calculation of function value"""
    return x * x - a
# End of f()

# Main
# Initial setting
xp = float(input("xp:"))
xn = float(input("xn:"))

# While Loop
while (xp - xn) * (xp - xn) > LIMIT:  
    xmid = (xp + xn) / 2              
    if f(xmid) > 0:                   
        xp = xmid                     
    else:                            
        xn = xmid                     
    print("{:.15f} {:.15f}".format(xn, xp))
# End of bisec.py 

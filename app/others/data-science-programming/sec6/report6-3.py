# -*- coding: utf-8 -*-
r"""
fracex.py
Demo for the fractions module
Usage　c:\>python fracex.py
"""
# Module import
from fractions import Fraction

# Main
print(Fraction(5, 10), Fraction(3, 15))                  # reduction of fraction
print(Fraction(1, 3) + Fraction(1, 7))                   # 1/3+1/7
print(Fraction(5, 3) * Fraction(6, 7) * Fraction(3, 2))  # 5/3*6/7*3/2
# End of the fracex.py 


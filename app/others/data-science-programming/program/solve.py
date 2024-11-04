# -*- coding: utf-8 -*-
"""
  solve.py
  Solve equations using the sympy module
Usage 　c:\>python solve.py
"""
#Module import
from sympy import *

#Main
var("x") 
equation=Eq(x**3+2*x**2-5*x-6,0) 
answer=solve(equation) 
print(answer) 

#End of solve.py

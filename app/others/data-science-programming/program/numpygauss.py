# -*- coding: utf-8 -*-
"""
  numpygauss.py
  Sample program for numpy module 
　Usage　C:\>python numpygauss.py
"""
#Module Import
import numpy as np

#Global Variables
a=np.array([[4,-1,0,-1,0,0,0,0,0],[-1,4,-1,0,-1,0,0,0,0],
   [0,-1,4,0,0,-1,0,0,0,],[-1,0,0,4,-1,0,-1,0,0],
   [0,-1,0,-1,4,-1,0,-1,0],[0,0,-1,0,-1,4,0,0,-1],
   [0,0,0,-1,0,0,4,-1,0],[0,0,0,0,-1,0,-1,4,-1],
   [0,0,0,0,0,-1,0,-1,4]])  
b=np.array([0,0,0.25,0,0,0.5,0.25,0.5,1.5]) 

#Main
x=np.linalg.solve(a,b) 
print(x)

# End of the numpygauss.py 



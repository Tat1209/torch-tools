#-*-coding: utf-8-*-
"""
gca1.py program
Cellular automaton (one-dimensional) calculation program 
Calculate time evolution from rules and initial state
Graph the results
Usage c: \> python gca1.py 
"""
# Import module
import sys # Necessary for using sys.exit ()
import numpy as np
import matplotlib.pyplot as plt

# Constant
N = 256 # Maximum number of cells
R = 8 # Rule table size
MAXT = 256 # number of iterations

# Define subcontract function
# setrule () function
def setrule (rule, ruleno):
    "" "Initialize rule table" ""
    # Write rule table
    for i in range (0, R):
        rule [i] = ruleno% 2
        ruleno = ruleno // 2 # shift left
    # Rule output
    for i in range (R-1, -1, -1):
        print (rule [i])
# end of setrule () function

# initca () function
def initca (ca):
    "" "Load initial values into cellular automaton" ""
    # Read initial value
    line = input ("Please enter the initial value of ca:")
    print ()
    #Conversion to internal representation
    for no in range (len (line)):
        ca [no] = int (line [no])
# end of initca () function

# putca () function
def putca (ca):
    "" "Ca status output" ""
    for no in range (N-1, -1, -1):
        print ("{: 1d}". format (ca [no]), end = "")
    print ()
# end of putca () function

# nextt () function
def nextt (ca, rule):
    "" "Ca status update" ""
    nextca = [0 for i in range (N)] # next generation ca
    # Apply rules
    for i in range (1, N-1):
        nextca [i] = rule [ca [i + 1] * 4 + ca [i] * 2 + ca [i-1]]
    # update ca
    for i in range (N):
        ca [i] = nextca [i]
# end of nextt () function

# Main execution part
outputdata = [[0 for i in range (N)] for j in range (MAXT + 1)]
# Initialize rules table
rule = [0 for i in range (R)] # Create rule table
ruleno = int (input ("Please enter the rule number:"))
if ruleno <0 or ruleno> 255:
        print ("Invalid rule number (", ruleno, ")")
        sys.exit ()
setrule (rule, ruleno) # set rule table
# Read initial value to cellular automaton
ca = [0 for i in range (N)] # cell array
initca (ca) # read initial value
putca (ca) # output ca status
for i in range (N):
    outputdata [0] [i] = ca [i]
# Time evolution calculation
for t in range (MAXT):
    nextt (ca, rule) # Update to next time
    putca (ca) # output ca status
    for i in range (N):
        outputdata [t + 1] [i] = ca [i]
# Graph output
plt.imshow (outputdata)
plt.show ()
# end of gca1.py

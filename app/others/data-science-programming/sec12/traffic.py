#-*-coding: utf-8-*-
"""
traffic.py program
Traffic flow simulation based on cellular automata
Calculate time evolution from rules and initial state
Usage c: \> python traffic.py <(initial state file name)
"""
# Import module
import sys # Necessary for using sys.exit ()

# Constant
N = 50 # Maximum number of cells
R = 8 # Rule table size
MAXT = 50 # number of iterations
RULE = 184 # Rule number (fixed to 184)

# Define subcontract function
# setrule () function
def setrule (rule, ruleno):
    "" "Initialize rule table" ""
    # Write rule table
    for i in range (0, R):
        rule [i] = ruleno% 2
        ruleno = ruleno // 2 # shift left
# end of setrule () function

# initca () function
def initca (ca):
    "" "Load initial values into cellular automaton" ""
    # Read initial value
    line = input ("Please enter the initial value of ca:")
    print ()
    # Convert to internal representation
    for no in range (len (line)):
        ca [N-1-no] = int (line [no])
# end of initca () function

# putca () function
def putca (ca):
    "" "Ca status output" ""
    for no in range (N-1, -1, -1):
        if ca [no] == 1:
            outputstr = "-"
        else:
            outputstr = ""
        print ("{:1s}". format (outputstr), end = "")
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
# Initialize inflow rate
flowrate = int (input ("Please enter the inflow rate:"))
print ()
if flowrate <= 0:
        print ("Inflow rate is incorrect (", flowrate, ")")
        sys.exit ()
# Initialize rules table
rule = [0 for i in range (R)] # Create rule table
setrule (rule, RULE) # set rule table
# Read initial value to cellular automaton
ca = [0 for i in range (N)] # cell array
initca (ca) # read initial value

# Time evolution calculation
for t in range (1, MAXT):
    nextt (ca, rule) # Update to next time
    if (t% flowrate) == 0:
         ca [N-2] = 1 # Car inflow
    print ("t =", t, "\t", end = "")     
    putca (ca) # output ca status
# end of traffic.py

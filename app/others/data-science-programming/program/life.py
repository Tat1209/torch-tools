#-*-coding: utf-8-*-
"""
life.py program
Life game calculation program 
A life game program, a type of two-dimensional cellular automaton
Usage c: \> python glife.py <(initial state file name)
The initial state file describes the initial layout
"""
# Import module
import sys # required for readlines ()

# Constant
N = 50 # Life game world size
MAXT = 10 # number of iterations

# Define subcontract function
# putworld () function
def putworld (world):
    "" "World status output" ""
    # update world
    for i in range (N):
        for j in range (N):
            print ("{: 1d}". format (world [i] [j]), end = "")
        print ()
# end of putworld () function

# initworld () function
def initworld (world):
    "" "Read initial value" ""
    chrworld = sys.stdin.readlines ()
    # Convert to internal representation
    for no, line in enumerate (chrworld):
        line = line.rstrip ()
        for i in range (len (line)):
            world [no] [i] = int (line [i])
# end of initworld () function

# nextt () function
def nextt (world):
    "" "Update state of the world" ""
    nextworld = [[0 for i in range (N)] for j in range (N)] # Next generation
    # Apply rules
    for i in range (1, N-1):
        for j in range (1, N-1):
            nextworld [i] [j] = calcnext (world, i, j)
    # update world
    for i in range (N):
        for j in range (N):
            world [i] [j] = nextworld [i] [j]
# end of nextt () function

# calcnext () function
def calcnext (world, i, j):
    "" "Update 1 cell status" ""
    no_of_one = 0 # Number of surrounding state 1 cells
    # Count cells in state 1
    for x in range (i-1, i + 2):
        for y in range (j-1, j + 2):
            no_of_one += world [x] [y]
    no_of_one-= world [i] [j] # don't count myself
    # Update state
    if no_of_one == 3:
        return 1 # birth
    elif no_of_one == 2:
        return world [i] [j] # survival
    return 0 # other than above
# end of calcnext () function

# Main execution part
world = [[0 for i in range (N)] for j in range (N)]
# load initial value to world [] []
initworld (world)
print ("t = 0") # Print initial time
putworld (world) # output world status

# Time evolution calculation
for t in range (1, MAXT):
    nextt (world) # update to next time
    print ("t =", t) # Print time
    putworld (world) # output world status
# end of life.py

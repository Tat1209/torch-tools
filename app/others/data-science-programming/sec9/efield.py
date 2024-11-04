#-*-coding: utf-8-*-
r"""
efield.py program
Simulation of 2D motion Simulation
of charged particles in electric field
Usage c: \> python efield.py
"""
# Import module
import math
# Constant
Q = (((0.0, 0.0), 10.0), ((5.0, -5.0), 5.0)) # Position and value of charge
TIMELIMIT = 20.0 # Simulation termination time
RLIMIT = 0.1 # Minimum value of distance r
H = 0.01 # Time step
# Main execution part
t = 0.0 # Time t
# Coefficient input
vx = float (input ("Please input initial speed v0x:"))
vy = float (input ("Please input initial speed v0y: "))
x = float (input ("Please enter the initial position x: "))
y = float (input ("Please enter the initial position y: "))
print ("{:.7f} {:.7f} {:.7f} {:.7f} {:.7f}".format (t, x, y, vx, vy))
# Current time and current position
# Calculate 2D motion
while t <TIMELIMIT: # Calculate until censoring time
    t = t + H # Update time
    rmin = float ("inf") # Initialize minimum distance
    for qi in Q:
        rx = qi [ 0] [0]-x # Calculate distance rx
        ry = qi [0] [1]-y # Calculate distance ry
        r = math.sqrt (rx * rx + ry * ry) # Calculate distance r
        if r < rmin:
            rmin = r # update minimum distance
        vx += (rx / r / r / r * qi [1]) * H # calculate velocity vx
        vy += (ry / r / r / r * qi [ 1]) * H # Calculation of velocity vy
        x += vx * H # Calculation of position x
        y += vy * H # Calculation of position y
        print ("{:.7f} {:.7f} {:.7f} {:.7f} {:.7f} ".format (t, x, y, vx, vy))
        # Current time and current position
        if rmin <RLIMIT:
            break # End if you are very close to the charge
# End of efield.py
# -*- coding: utf-8 -*-
"""
infection.py program
Agent simulation of “infection”
Agents operating in a 2D plane
Two types of agents interact
Usage c: \> python infection.py
"""
# Import module
import random
import numpy as np
import matplotlib.pyplot as plt

# Global variable
N = 100 # number of agents
TIMELIMIT = 100 # Simulation stop time
SEED = 65535 # random seed
R = 0.5 # A number that defines the neighborhood
factor = 1.0 # Category 1 agent stride

# Class definition
# Agent class
class Agent:
    """Definition of a class representing an agent """
    def __init__ (self, cat): # constructor
        self.category = cat
        self.x = 0 # initial x coordinate
        self.y = 0 # initial y coordinate
    def calcnext (self): # next state calculation
        if self.category == 0:
            self.cat0 () # Category 0 calculation
        elif self.category == 1:
            self.cat1 () # Category 1 calculation
        else: # no matching category
            print ("ERROR category missing \ n") 
    def cat0 (self): # Category 0 calculation method
        # Find distance to Category 1 agent
        for i in range(len(a)):
            if a[i].category == 1:
                c0x = self.x
                c0y = self.y 
                ax = a[i].x 
                ay = a [i] .y
                if ((c0x - ax) * (c0x - ax) + (c0y - ay) * (c0y - ay)) < R:
                # Adjacent category 1 agent
                    self.category = 1 # transforms into category 1
            else: # Category 1 is not in the neighborhood
                self.x += random.random() - 0.5
                self.y += random.random() - 0.5

    def cat1 (self): # Category 1 calculation method
        self.x += (random.random() - 0.5) * factor
        self.y += (random.random() - 0.5) * factor
    def putstate (self): # output state
        print(self.category, self.x, self.y)
# end of agent class definition

# Define subcontract function
# calcn () function
def calcn(a):
    "" "Calculate next time status" ""
    for i in range(len(a)):
        a[i].calcnext()
        a[i].putstate()
# end of calcn () function

# Main execution part
# Initialization
random.seed (SEED) # random number initialization
# Create a category 0 agent
a = [Agent(0) for i in range(N)]
# Set up category 1 agent
a[0].category = 1
a[0].x = -2 
a [0] .y = -2 
# Set step factor for category 1 agent
factor = float (input ("Enter the stride factor for category 1"))

# Agent simulation
for t in range(TIMELIMIT):
    print("t=", t)
    calcn (a) # calculate next time state
# end of infection.py

import irsim
import numpy as np
from PIL import Image
#from irsim.env import add_object
import cv2

env = irsim.make('maps/map1.yaml') # initialize the environment with the configuration file
env.load_behavior("agent_behavior")

for i in range(1500): # run the simulation for 300 steps
    #print(f'step #{i}')
    env.step()  # update the environment
    env.render() # render the environment
    #input()
    if env.done(): 
        break
env.end() # close the environment


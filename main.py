import irsim

env = irsim.make('maps/map3.yaml') # initialize the environment with the configuration file

#env.load_behavior("orca_behavior") 
env.load_behavior("agent_behavior")

for i in range(800): # run the simulation for 300 steps

    env.step()  # update the environment
    env.render() # render the environment

    if env.done(): 
        break # check if the simulation is done
        
env.end() # close the environment
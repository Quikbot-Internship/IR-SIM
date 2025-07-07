import irsim

env = irsim.make('maps/map2.yaml') # initialize the environment with the configuration file
env.load_behavior("agent_behavior")

for i in range(1500): # run the simulation for 300 steps

    env.step()  # update the environment
    env.render() # render the environment

    if env.done(): 
        break
env.end() # close the environment


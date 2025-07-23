import irsim
import numpy as np
from simulation_globals import robot_states_last_step

env = irsim.make('maps/map4.yaml') # initialize the environment with the configuration file
env.load_behavior("agent_behavior")

for i in range(1500): # run the simulation for 300 steps
    print(f'step #{i}')

    # Phase 1: Snapshot all robot states
    robot_states_last_step.clear()
    for robot in env.robot_list:
        robot_states_last_step[robot.name] = {
            'position': robot.state[:2].copy(),
            'velocity': getattr(robot, 'velocity', np.zeros((2,1))).copy(),
            'heading': robot.state[2, 0],
        }

    env.step()  # update the environment
    env.render() # render the environment
    input()
    if env.done(): 
        break
env.end() # close the environment


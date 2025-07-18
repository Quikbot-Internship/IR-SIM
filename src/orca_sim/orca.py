from orca_sim.RVOSimulator import RVOSimulator
from orca_sim.Vector2 import Vector2
import numpy as np
from orca_sim.Vector2 import Vector2
from avoidance.orca_utils import compute_pref_velocity
from irsim.util.util import WrapToPi, omni_to_diff
from simulation_globals import robot_states_last_step

class ORCA_RVOPlanner:
    def __init__(self, ego_object, external_objects, time_horizon):
        #self.robots = robots
        self.sim = RVOSimulator(timeStep=0.1)
        self.agent_ids = []
        self.ego_object = ego_object
        self.external_objects = external_objects
        self.time_horizon = time_horizon
        self.time_step = 0.1

        # Set agent defaults
        self.sim.setAgentDefaults(
            neighborDist=5.0,
            maxNeighbors=10,
            timeHorizon=time_horizon,
            timeHorizonObst=5.0,
            radius=0.4,
            maxSpeed=1.0,
            velocity=Vector2(0.0, 0.0)
        )

    def compute_control(self, goal):
        #print("Computing control for ORCA_RVO Planner")
        self.sim.clear()  # Clear previous state

        goal = goal
        _, max_vel = self.ego_object.get_vel_range()
        max_linear_vel = max_vel[0, 0]
        max_angular_vel = max_vel[1, 0]
        robot_radius = self.ego_object.radius

        # Build preferred velocity
        pos = np.array(self.ego_object.state[:2]).flatten()
        heading = self.ego_object.state[2, 0]
        vel = np.array([self.ego_object.velocity[0, 0], self.ego_object.velocity[1, 0]])

        # Add ego agent
        ego_id = self.sim.addAgent(
            position=Vector2(*pos),
            velocity=Vector2(*vel),
            neighborDist=robot_radius * 5.0,
            maxNeighbors=10,
            timeHorizon=self.time_horizon,
            timeHorizonObst=self.time_horizon,
            radius=robot_radius,
            maxSpeed=max_linear_vel
        )

        # Compute preferred velocity toward goal
        pref_vel = compute_pref_velocity(pos, goal, max_linear_vel)
        pref_vel_vec = Vector2(pref_vel[0], pref_vel[1])
        self.sim.setAgentPrefVelocity(ego_id, pref_vel_vec)

        # Add surrounding agents (other robots)
        for obj in self.external_objects:
            if obj.name == self.ego_object.name:
                continue

            # Get the last snapshot of the robot state
            # This assumes robot_states_last_step is a dictionary with robot names as keys
            last = robot_states_last_step.get(obj.name)
            if last is None:
                continue  # no snapshot available

            other_pos = np.array(last['position']).flatten()
            other_vel = np.array(last['velocity']).flatten()

            self.sim.addAgent(
                position=Vector2(*other_pos),
                velocity=Vector2(*other_vel),
                neighborDist=obj.radius * 5.0,        # or customize as needed
                maxNeighbors=10,
                timeHorizon=self.time_horizon,
                timeHorizonObst=self.time_horizon,
                radius=obj.radius,
                maxSpeed=max_linear_vel
            )

        # Perform ORCA velocity planning (step computes new velocities)
        self.sim.doStep()

        # Get ego's new velocity
        new_vel = self.sim.getAgentVelocity(ego_id)

        if self.ego_object.color == 'g':
            print(f'preferred velocity: {self.sim.getAgentPrefVelocity(ego_id)}')
            print(f'orca velocity: {[new_vel.x_, new_vel.y_]}\n')

        return omni_to_diff(heading, [new_vel.x_, new_vel.y_], max_angular_vel)
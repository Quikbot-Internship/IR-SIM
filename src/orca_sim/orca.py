from orca_sim.RVOSimulator import RVOSimulator
from orca_sim.Vector2 import Vector2
import numpy as np
from orca_sim.Vector2 import Vector2
from orca_sim.orca_utils import compute_pref_velocity
from irsim.util.util import WrapToPi, omni_to_diff, diff_to_omni
from simulation_globals import robot_states_last_step

class ORCA_RVOPlanner:
    def __init__(self, ego_object, external_objects, time_horizon):
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
            timeHorizonObst=time_horizon,
            radius=0.6,
            maxSpeed=1.0,
            velocity=Vector2(0.0, 0.0)
        )

    def compute_control(self, goal):
        self.sim.clear()  # Clear previous state

        robot_name = ''
        if self.ego_object.color == 'g':
            robot_name = 'Green-robot'
        elif self.ego_object.color == 'r':
            robot_name = 'Red-robot'
        elif self.ego_object.color == 'b':
            robot_name = 'Blue-robot'

        _, max_vel = self.ego_object.get_vel_range()
        max_linear_vel = max_vel[0, 0]
        max_angular_vel = max_vel[1, 0]
        robot_radius = self.ego_object.radius

        # Use last step snapshot for centralized consistency
        last_state = robot_states_last_step[self.ego_object.name]
        pos = np.array(last_state["position"]).flatten()
        heading = last_state["heading"]
        vel_diff = np.array(last_state["velocity"]).reshape((2, 1))
        vel = diff_to_omni(heading, vel_diff)  # Convert to omni


        # Add ego agent to ORCA sim using last known data
        ego_id = self.sim.addAgent(
            position=Vector2(*pos),
            velocity=Vector2(vel[0, 0], vel[1, 0]),
            neighborDist=robot_radius * 20.0,
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
            if not obj.name.startswith('robot'):
                continue  # Skip other robots in this planner
            
            # Get the last snapshot of the robot state
            last = robot_states_last_step.get(obj.name)
            if last is None:
                continue  

            # Use last known position, heading, and velocity
            other_pos = np.array(last['position']).flatten()
            other_vel_diff = np.array(last['velocity']).reshape((2, 1))
            other_heading = last['heading']
            other_vel = diff_to_omni(other_heading, other_vel_diff)  

            agent_idx = self.sim.addAgent(
                position=Vector2(other_pos[0], other_pos[1]),
                velocity=Vector2(other_vel[0, 0], other_vel[1, 0]),
                neighborDist=obj.radius * 20.0,        
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

        print(f'{robot_name} preferred velocity: {self.sim.getAgentPrefVelocity(ego_id)}')
        print(f'{robot_name} orca velocity: {[new_vel.x_, new_vel.y_]}\n')
        

        return omni_to_diff(heading, [new_vel.x_, new_vel.y_], max_angular_vel)
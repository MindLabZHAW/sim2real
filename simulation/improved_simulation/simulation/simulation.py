import random
import time
from assets.assetFactory import AssetFactory
from config.config import Configuration
from simulation.dataProcessor import DataProcessor
from simulation.movement.ForcedContactMovement import ForcedContactMovement
from utils import utils
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *

class Simulation:
    def __init__(self, gym, sim, viewer, device, sim_data):
        self.gym = gym
        self.sim = sim
        self.viewer = viewer
        self.device = device
        self.sim_data = sim_data
        self.data_dict = []
        self.movemenet = ForcedContactMovement(self.sim_data.init_pos)

    def run(self, duration_time):
        # simulation loop
        start_time = time.time()

        dataProcessor = DataProcessor(self.sim_data, start_time)
        dataProcessor.cleanup()

        index_number = 0
        while time.time() - start_time < duration_time:  # not gym.query_viewer_has_closed(viewer):
            self.tick(start_time, index_number)
            dataProcessor.process()
            index_number += 1

        dataProcessor.save_data()

    def tick(self, start_time, index_number):
        #time.sleep(0.1)
        self.step_physics()
        self.refresh_tensors()        
        self.detect_contact()

        coordinates_for_new_position_tensor = self.movemenet.get_coordinates_for_next_position(self.sim_data.rb_states, index_number)

        # compute position and orientation error
        position_and_orientation_error_tensor = torch.cat([self.get_position_error_tensor(coordinates_for_new_position_tensor), self.get_orientation_error_tensor()], -1).unsqueeze(-1)
 
         # Deploy control based on type
        if self.sim_data.controller == "ik":
            self.sim_data.pos_action[:, :7] = self.get_new_position_action(position_and_orientation_error_tensor)
        else:       # osc
            self.sim_data.effort_action[:, :7] = self.get_new_effort_action(position_and_orientation_error_tensor)    

        self.deploy_pos_and_effort_action_to_franka_robot()
        self.update_viewer()

    def get_orientation_error_tensor(self):
        goal_rotation = quat_mul(self.sim_data.down_q, quat_conjugate(self.get_cube_grasping_yaw()))
        return utils.orientation_error(goal_rotation, self.get_hand_rotation())

    def get_position_error_tensor(self, grasping_position):
        goal_position = grasping_position
        position_error = goal_position - self.get_hand_position()
        return position_error

    def get_cube_grasping_yaw(self):
        box_rotation = self.sim_data.rb_states[self.sim_data.box_idxs, 3:7]
        return utils.cube_grasping_yaw(box_rotation, self.sim_data.corners)

    def get_hand_rotation(self):
        return self.sim_data.rb_states[self.sim_data.hand_idxs, 3:7]
    
    def get_hand_position(self):
        return self.sim_data.rb_states[self.sim_data.hand_idxs, :3]

    def detect_contact(self):
        #contact detection
        #print(net_cf)
        #print(panda_idxs)
        #print(net_cf[panda_idxs])
        if self.has_contact():
            print('There is a Contact :(') 
            print(self.sim_data.net_cf[self.sim_data.panda_idxs])
        else:
            print('There is no Contact :)')
            #print(contacted_link)
        #control

    def deploy_pos_and_effort_action_to_franka_robot(self):
        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.sim_data.pos_action))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.sim_data.effort_action))


    def update_viewer(self):
         # update viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)

    def step_physics(self):
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def has_contact(self):
        contacted_link = utils.count_nonzero(self.sim_data.net_cf[self.sim_data.panda_idxs])
        return torch.count_nonzero(contacted_link) > 0


    def refresh_tensors(self):
        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
            
    def get_new_position_action(self, position_and_orientation_error_tensor):
        return self.sim_data.dof_pos.squeeze(-1)[:, :7] + utils.control_ik(position_and_orientation_error_tensor, self.sim_data.j_eef, self.device, Configuration.DAMPING, self.sim_data.num_envs)
    
    def get_new_effort_action(self, position_and_orientation_error_tensor):
        hand_vel = self.sim_data.rb_states[self.sim_data.hand_idxs, 7:]   
        return utils.control_osc(position_and_orientation_error_tensor, self.sim_data.mm, self.sim_data.j_eef, Configuration.KP, Configuration.KP_NULL, Configuration.KD, Configuration.KD_NULL,
                                                                    hand_vel, self.sim_data.dof_vel, self.sim_data.default_dof_pos_tensor, self.sim_data.dof_pos, self.device)

        

            
            
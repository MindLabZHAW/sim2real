import time
from assets.assetFactory import AssetFactory
from config.config import Configuration
from simulation.dataProcessor import DataProcessor
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

        # if hand is above box, descend to grasp offset
        # otherwise, seek a position above the box
        grasping_position = self.get_box_position().clone()
        grasping_position[:, 2] = torch.where(self.is_hand_above_box(), self.get_box_position()[:, 2] + self.get_grasping_offset(), self.get_box_position()[:, 2] + self.get_grasping_offset() * 2.5)

        # compute position and orientation error
        position_and_orientation_error_tensor = torch.cat([self.get_position_error_tensor(grasping_position), self.get_orientation_error_tensor()], -1).unsqueeze(-1)
 
         # Deploy control based on type
        if self.sim_data.controller == "ik":
            self.sim_data.pos_action[:, :7] = self.get_new_position_action(position_and_orientation_error_tensor)
        else:       # osc
            self.sim_data.effort_action[:, :7] = self.get_new_effort_action(position_and_orientation_error_tensor)    

        self.deploy_pos_and_effort_action_to_franka_robot()
        self.update_viewer()

    def get_orientation_error_tensor(self):
        goal_rotation = torch.where(self.should_hand_return_to_start_position(), self.sim_data.init_rot, quat_mul(self.sim_data.down_q, quat_conjugate(self.get_cube_grasping_yaw())))
        return utils.orientation_error(goal_rotation, self.get_hand_rotation())

    def get_position_error_tensor(self, grasping_position):
        goal_position = torch.where(self.should_hand_return_to_start_position(), self.sim_data.init_pos, grasping_position)
        position_error = goal_position - self.get_hand_position()
        return position_error
        

    def should_gripper_stay_closed(self):
        # always open the gripper above a certain height, dropping the box and restarting from the beginning
        keep_going = torch.logical_not(self.is_initial_hand_position_reached() | (self.get_box_position()[:, 2] > 0.6))
        should_gripper_be_closed = (self.get_distance_from_hand_to_box() < self.get_grasping_offset() + 0.02) | self.is_gripper_holding_box(self.get_distance_from_hand_to_box(), self.get_grasping_offset())
        return should_gripper_be_closed & keep_going.unsqueeze(-1)

    def is_hand_above_box(self):
        box_yaw_direction = utils.quat_axis(self.get_cube_grasping_yaw(), 0)
        hand_yaw_direction = utils.quat_axis(self.get_hand_rotation(), 0)
        yaw_dot = torch.bmm(box_yaw_direction.view(self.sim_data.num_envs, 1, 3), hand_yaw_direction.view(self.sim_data.num_envs, 3, 1)).squeeze(-1)
        box_direction = self.get_vector_from_hand_to_box() / self.get_distance_from_hand_to_box()
        box_dot = box_direction @ self.sim_data.down_dir.view(3, 1)
        return ((box_dot >= 0.99) & (yaw_dot >= 0.95) & (self.get_distance_from_hand_to_box() < self.get_grasping_offset() * 3)).squeeze(-1)


    def get_cube_grasping_yaw(self):
        box_rotation = self.sim_data.rb_states[self.sim_data.box_idxs, 3:7]
        return utils.cube_grasping_yaw(box_rotation, self.sim_data.corners)

    def should_hand_return_to_start_position(self):
         # if initial hand position reached; allow the hand to start moving to the box
        return (self.is_initial_hand_position_reached() | self.is_gripper_holding_box(self.get_distance_from_hand_to_box(), self.get_grasping_offset()).squeeze(-1)).unsqueeze(-1)

    def is_initial_hand_position_reached(self):
        # determine if we have reached the initial position; 
        return (self.sim_data.hand_restart & (self.get_distance_from_hand_position_to_initial_position() > 0.02)).squeeze(-1)
    
    def get_grasping_offset(self):
        # how far the hand should be from box for grasping
        return 0.11 if self.sim_data.controller == "ik" else 0.10

    def get_hand_rotation(self):
        return self.sim_data.rb_states[self.sim_data.hand_idxs, 3:7]

    def get_distance_from_hand_position_to_initial_position(self):
        return torch.norm(self.get_vector_from_hand_position_to_initial_position(), dim=-1)

    def get_vector_from_hand_position_to_initial_position(self):
        return self.sim_data.init_pos - self.get_hand_position()

    def get_distance_from_hand_to_box(self):
        return torch.norm(self.get_vector_from_hand_to_box(), dim=-1).unsqueeze(-1)

    def get_vector_from_hand_to_box(self):
        return self.get_box_position() - self.get_hand_position()

    def get_box_position(self):
        return self.sim_data.rb_states[self.sim_data.box_idxs, :3]

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

    def is_gripper_holding_box(self, box_dist, grasp_offset):
        # determine if we're holding the box (grippers are closed and box is near)
        gripper_sep = self.sim_data.dof_pos[:, 7] + self.sim_data.dof_pos[:, 8]
        return (gripper_sep < 0.045) & (box_dist < grasp_offset + 0.5 * AssetFactory.BOX_SIZE) #true or false

    def deploy_pos_and_effort_action_to_franka_robot(self):
         # gripper actions depend on distance between hand and box
        should_gripper_stay_closed = self.should_gripper_stay_closed()
        gripper_actions = torch.where(should_gripper_stay_closed, torch.Tensor([[0., 0.]] * self.sim_data.num_envs).to(self.device), torch.Tensor([[0.04, 0.04]] * self.sim_data.num_envs).to(self.device))
        self.sim_data.pos_action[:, 7:9] = gripper_actions

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

        

            
            
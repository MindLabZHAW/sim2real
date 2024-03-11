import time
from assets.assetFactory import AssetFactory
from config.config import Configuration
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

        index_number = 0
        while time.time() - start_time < duration_time:  # not gym.query_viewer_has_closed(viewer):
            self.tick(start_time, index_number)
            index_number += 1


    def tick(self, start_time, index_number):
        #time.sleep(0.1)
        self.step_physics()
        self.refresh_tensors()        
        self.detect_contact()
    
        # determine if we have reached the initial position; if so allow the hand to start moving to the box
        # how far the hand should be from box for grasping
        grasp_offset = 0.11 if self.sim_data.controller == "ik" else 0.10
        hand_restart = (self.sim_data.hand_restart & (self.get_distance_from_hand_position_to_initial_position() > 0.02)).squeeze(-1)
        return_to_start = (hand_restart | self.is_gripper_holding_box(self.get_distance_from_hand_to_box(), grasp_offset).squeeze(-1)).unsqueeze(-1)

        # if hand is above box, descend to grasp offset
        # otherwise, seek a position above the box
        box_rot = self.sim_data.rb_states[self.sim_data.box_idxs, 3:7]
        yaw_q = utils.cube_grasping_yaw(box_rot, self.sim_data.corners)
        box_yaw_dir = utils.quat_axis(yaw_q, 0)
        hand_rot = self.sim_data.rb_states[self.sim_data.hand_idxs, 3:7]
        hand_yaw_dir = utils.quat_axis(hand_rot, 0)
        yaw_dot = torch.bmm(box_yaw_dir.view(self.sim_data.num_envs, 1, 3), hand_yaw_dir.view(self.sim_data.num_envs, 3, 1)).squeeze(-1)
        box_dir = self.get_vector_from_hand_to_box() / self.get_distance_from_hand_to_box()
        box_dot = box_dir @ self.sim_data.down_dir.view(3, 1)
        above_box = ((box_dot >= 0.99) & (yaw_dot >= 0.95) & (self.get_distance_from_hand_to_box() < grasp_offset * 3)).squeeze(-1)
        grasp_pos = self.get_box_position().clone()
        grasp_pos[:, 2] = torch.where(above_box, self.get_box_position()[:, 2] + grasp_offset, self.get_box_position()[:, 2] + grasp_offset * 2.5)

        # compute goal position and orientation
        goal_pos = torch.where(return_to_start, self.sim_data.init_pos, grasp_pos)
        goal_rot = torch.where(return_to_start, self.sim_data.init_rot, quat_mul(self.sim_data.down_q, quat_conjugate(yaw_q)))

        # compute position and orientation error
        pos_err = goal_pos - self.get_hand_position()
        orn_err = utils.orientation_error(goal_rot, hand_rot)
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
 
        self.deploy_control(dpose)
        
        # always open the gripper above a certain height, dropping the box and restarting from the beginning
        hand_restart = hand_restart | (self.get_box_position()[:, 2] > 0.6)
        keep_going = torch.logical_not(hand_restart)
        # gripper actions depend on distance between hand and box
        close_gripper = (self.get_distance_from_hand_to_box() < grasp_offset + 0.02) | self.is_gripper_holding_box(self.get_distance_from_hand_to_box(), grasp_offset)
        close_gripper = close_gripper & keep_going.unsqueeze(-1)
        grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * self.sim_data.num_envs).to(self.device), torch.Tensor([[0.04, 0.04]] * self.sim_data.num_envs).to(self.device))
        self.sim_data.pos_action[:, 7:9] = grip_acts

        self.deploy_actions()
        self.update_viewer()
        self.save_data_to_dict(index_number, start_time)

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

    def deploy_actions(self):
        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.sim_data.pos_action))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.sim_data.effort_action))


    def save_data_to_dict(self, index_number, start_time):
        if index_number == 0:
            self.data_dict = { 'time':  [time.time()-start_time], 'contact' : self.sim_data.net_cf[self.sim_data.panda_idxs][None,:,:]}
        self.data_dict['time'] = np.append(self.data_dict['time'],[time.time()-start_time])
        self.data_dict['contact'] = torch.cat((self.data_dict['contact'] , self.sim_data.net_cf[self.sim_data.panda_idxs][None,:,:]))

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


    def deploy_control(self, dpose):
        hand_vel = self.sim_data.rb_states[self.sim_data.hand_idxs, 7:]   
         # Deploy control based on type
        if self.sim_data.controller == "ik":
            self.sim_data.pos_action[:, :7] = self.sim_data.dof_pos.squeeze(-1)[:, :7] + utils.control_ik(dpose, self.sim_data.j_eef, self.device, Configuration.DAMPING, self.sim_data.num_envs)
        else:       # osc
            self.sim_data.effort_action[:, :7] = utils.control_osc(dpose, self.sim_data.mm, self.sim_data.j_eef, Configuration.KP, Configuration.KP_NULL, Configuration.KD, Configuration.KD_NULL,
                                                                    hand_vel, self.sim_data.dof_vel, self.sim_data.default_dof_pos_tensor, self.sim_data.dof_pos, self.device)
        
    def get_data_dict(self):
        return self.data_dict.copy()
            
            
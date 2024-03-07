import time
from assets.assetFactory import AssetFactory
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
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        #contact detection
        #print(net_cf)
        #print(panda_idxs)
        #print(net_cf[panda_idxs])
        contacted_link = utils.count_nonzero(self.sim_data.net_cf[self.sim_data.panda_idxs])
        if torch.count_nonzero(contacted_link) == 0:
            print('There is no Contact :)')
        else:
            print('There is a Contact :(')
            #print(contacted_link)
            print(self.sim_data.net_cf[self.sim_data.panda_idxs])
        #control
        
        box_pos = self.sim_data.rb_states[self.sim_data.box_idxs, :3]
        box_rot = self.sim_data.rb_states[self.sim_data.box_idxs, 3:7]

        hand_pos = self.sim_data.rb_states[self.sim_data.hand_idxs, :3]
        hand_rot = self.sim_data.rb_states[self.sim_data.hand_idxs, 3:7]
        hand_vel = self.sim_data.rb_states[self.sim_data.hand_idxs, 7:]

        to_box = box_pos - hand_pos
        box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
        box_dir = to_box / box_dist
        box_dot = box_dir @ self.sim_data.down_dir.view(3, 1)

        # how far the hand should be from box for grasping
        grasp_offset = 0.11 if self.sim_data.controller == "ik" else 0.10

        # determine if we're holding the box (grippers are closed and box is near)
        gripper_sep = self.sim_data.dof_pos[:, 7] + self.sim_data.dof_pos[:, 8]
        gripped = (gripper_sep < 0.045) & (box_dist < grasp_offset + 0.5 * AssetFactory.BOX_SIZE) #true or false

        yaw_q = utils.cube_grasping_yaw(box_rot, self.sim_data.corners)
        box_yaw_dir = utils.quat_axis(yaw_q, 0)
        hand_yaw_dir = utils.quat_axis(hand_rot, 0)
        yaw_dot = torch.bmm(box_yaw_dir.view(self.sim_data.num_envs, 1, 3), hand_yaw_dir.view(self.sim_data.num_envs, 3, 1)).squeeze(-1)

        # determine if we have reached the initial position; if so allow the hand to start moving to the box
        to_init = self.sim_data.init_pos - hand_pos
        init_dist = torch.norm(to_init, dim=-1)
        hand_restart = (self.sim_data.hand_restart & (init_dist > 0.02)).squeeze(-1)
        return_to_start = (hand_restart | gripped.squeeze(-1)).unsqueeze(-1)

        # if hand is above box, descend to grasp offset
        # otherwise, seek a position above the box
        above_box = ((box_dot >= 0.99) & (yaw_dot >= 0.95) & (box_dist < grasp_offset * 3)).squeeze(-1)
        grasp_pos = box_pos.clone()
        grasp_pos[:, 2] = torch.where(above_box, box_pos[:, 2] + grasp_offset, box_pos[:, 2] + grasp_offset * 2.5)

        # compute goal position and orientation
        goal_pos = torch.where(return_to_start, self.sim_data.init_pos, grasp_pos)
        goal_rot = torch.where(return_to_start, self.sim_data.init_rot, quat_mul(self.sim_data.down_q, quat_conjugate(yaw_q)))

        # compute position and orientation error
        pos_err = goal_pos - hand_pos
        orn_err = utils.orientation_error(goal_rot, hand_rot)
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

        # Deploy control based on type
        if self.sim_data.controller == "ik":
            self.sim_data.pos_action[:, :7] = self.sim_data.dof_pos.squeeze(-1)[:, :7] + utils.control_ik(dpose, self.sim_data.j_eef, self.device, self.sim_data.damping, self.sim_data.num_envs)
        else:       # osc
            self.sim_data.effort_action[:, :7] = utils.control_osc(dpose, self.sim_data.mm, self.sim_data.j_eef, self.sim_data.kp, self.sim_data.kp_null, self.sim_data.kd, self.sim_data.kd_null,
                                                                    hand_vel, self.sim_data.dof_vel, self.sim_data.default_dof_pos_tensor, self.sim_data.dof_pos, self.device)

        # gripper actions depend on distance between hand and box
        close_gripper = (box_dist < grasp_offset + 0.02) | gripped
        # always open the gripper above a certain height, dropping the box and restarting from the beginning
        hand_restart = hand_restart | (box_pos[:, 2] > 0.6)
        keep_going = torch.logical_not(hand_restart)
        close_gripper = close_gripper & keep_going.unsqueeze(-1)
        grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * self.sim_data.num_envs).to(self.device), torch.Tensor([[0.04, 0.04]] * self.sim_data.num_envs).to(self.device))
        self.sim_data.pos_action[:, 7:9] = grip_acts

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.sim_data.pos_action))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.sim_data.effort_action))

        # update viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)
        if index_number == 0:
            self.data_dict = { 'time':  [time.time()-start_time], 'contact' : self.sim_data.net_cf[self.sim_data.panda_idxs][None,:,:]}
        self.data_dict['time'] = np.append(self.data_dict['time'],[time.time()-start_time])
        self.data_dict['contact'] = torch.cat((self.data_dict['contact'] , self.sim_data.net_cf[self.sim_data.panda_idxs][None,:,:]))
        
    def get_data_dict(self):
        return self.get_data_dict.copy()
            
            
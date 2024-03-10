import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from simulation.improved_simulation.assets.assetFactory import AssetFactory

class TensorDataProcessor:
    def __init__(self, gym, sim, num_envs, init_pos_list, init_rot_list, default_dof_pos, panda_idxs):
        self.gym = gym
        self.sim = sim
        self.num_envs = num_envs
        self.init_pos_list = init_pos_list
        self.init_rot_list = init_rot_list
        self.default_dof_pos_tensor = default_dof_pos
        self.panda_idxs = panda_idxs

        self.init_pos = None
        self.init_rot = None
        self.down_q = None
        self.down_dir = None
        self.jacobian = None
        self.mass_matrix = None
        self.rb_states = None
        self.net_contact_force = None
        self.hand_restart = None
        self.corners = None
        self.dof_pos = None
        self.dof_vel = None
        self.pos_action = None
        self.effort_action = None
        self.contacted_link = None

    def process_tensor_data(self, device):
        self.init_pos_and_rot_tensor(device)
        self.init_down_q_tensor(device)
        self.init_down_dir_tensor(device)
        self.init_jacobian_tensor()
        self.init_mass_matrix_tensor()
        self.init_rigid_body_state_tensor()
        self.init_net_contact_force_tensor()
        self.init_hand_restart_tensor(device)
        self.init_box_corners_tensor(device)
        self.init_dof_pos_tensor()
        self.init_dof_vel_tensor()
        self.init_pos_action_tensor()
        self.init_effort_action_tensor()
        self.init_effort_action_tensor()
        self.init_default_dof_pos_tensor(device)
        self.init_contacted_link_tensor()

    def init_contacted_link_tensor(self):
        self.contacted_link =torch.zeros_like(torch.tensor(self.panda_idxs))

    def init_default_dof_pos_tensor(self, device):
        # send to torch
        self.default_dof_pos_tensor = to_torch(self.default_dof_pos_tensor, device=device)

    def init_effort_action_tensor(self):
        # Set action tensors
        self.effort_action = torch.zeros_like(self.pos_action)

    def init_pos_action_tensor(self):
        # Set action tensors
        self.pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)

    def init_dof_vel_tensor(self):
        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_vel = dof_states[:, 1].view(self.num_envs, 9, 1)
        
    def init_dof_pos_tensor(self):
        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_pos = dof_states[:, 0].view(self.num_envs, 9, 1)

    def init_box_corners_tensor(self, device):
        # box corner coords, used to determine grasping yaw
        box_half_size = 0.5 * AssetFactory.BOX_SIZE
        corner_coord = torch.Tensor([box_half_size, box_half_size, box_half_size])
        self.corners = torch.stack(self.num_envs * [corner_coord]).to(device)

    def init_hand_restart_tensor(self, device):
        # Create a tensor noting whether the hand should return to the initial position
        self.hand_restart = torch.full([self.num_envs], False, dtype=torch.bool).to(device)

    def init_net_contact_force_tensor(self):
        # get net contactforse tensor
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.net_contact_force = gymtorch.wrap_tensor(_net_cf)

    def init_rigid_body_state_tensor(self):
        # get rigid body state tensor__pycache__/**/*
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

    def init_mass_matrix_tensor(self):
        # get mass matrix tensor
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        self.mass_matrix = gymtorch.wrap_tensor(_massmatrix)
        self.mass_matrix = self.mass_matrix[:, :7, :7]          # only need elements corresponding to the franka arm

    def init_jacobian_tensor(self):
        # get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        self.jacobian = gymtorch.wrap_tensor(_jacobian)

    def init_down_dir_tensor(self, device):
        # downard axis
        self.down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

    def init_pos_and_rot_tensor(self, device):
        # initial hand position and orientation tensors
        self.init_pos = torch.Tensor(self.init_pos_list).view(self.num_envs, 3).to(device)
        self.init_rot = torch.Tensor(self.init_rot_list).view(self.num_envs, 4).to(device)

    def init_down_q_tensor(self, device):
        # hand orientation for grasping
        self.down_q = torch.stack(self.num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((self.num_envs, 4))

    def get_init_pos_tensor(self):
        return self.init_pos
    
    def get_init_rot_tensor(self):
        return self.init_rot
    
    def get_down_q_tensor(self):
        return self.down_q
    
    def get_down_dir_tensor(self):
        return self.down_dir
    
    def get_jacobian_tensor(self):
        return self.jacobian
    
    def get_mass_matrix_tensor(self):
        return self.mass_matrix
    
    def get_rigib_body_states_tensor(self):
        return self.rb_states
    
    def get_net_contact_force_tensor(self):
        return self.net_contact_force
    
    def get_hand_restart_tensor(self):
        return self.hand_restart
    
    def get_box_corners_tensor(self):
        return self.corners
    
    def get_dof_vel_tensor(self):
        return self.dof_vel
    
    def get_dof_pos_tensor(self):
        return self.dof_pos
    
    def get_pos_action_tensor(self):
        return self.pos_action

    def get_effort_action_tensor(self):
        return self.effort_action

    def get_default_pos_tensor(self):
        return self.default_dof_pos_tensor
    
    def get_contacted_link_tensor(self):
        return self.contacted_link

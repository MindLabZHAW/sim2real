from isaacgym import gymapi

import numpy as np

class DofDataProcessor:
    def __init__(self, gym, franka_asset):
        self.gym = gym
        self.franka_asset = franka_asset
        self.franka_num_dofs = self.gym.get_asset_dof_count(self.franka_asset)
        self.franka_lower_limits = None
        self.franka_upper_limits = None
        self.default_dof_pos = None
        self.default_dof_state = None
        self.franka_dof_props = None

        

    def process_dof_data(self, controller):
        self.set_franka_limits()
        self.init_franka_dof_props(controller)
        self.init_default_dof_states_and_position_target()
        self.init_default_dof_grippers_open()
        self.init_default_dof_state()


    def set_franka_limits(self):
        self.franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        self.franka_lower_limits = self.franka_dof_props["lower"]
        self.franka_upper_limits = self.franka_dof_props["upper"]


    def init_franka_dof_props(self, controller):
        # use position drive for all dofs
        if controller == "ik":
            self.franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
            self.franka_dof_props["stiffness"][:7].fill(400.0)
            self.franka_dof_props["damping"][:7].fill(40.0)
        else:       # osc
            self.franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
            self.franka_dof_props["stiffness"][:7].fill(0.0)
            self.franka_dof_props["damping"][:7].fill(0.0)
        # grippers
        self.franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        self.franka_dof_props["stiffness"][7:].fill(800.0)
        self.franka_dof_props["damping"][7:].fill(40.0)

    def init_default_dof_states_and_position_target(self):
        # default dof states and position targets
        franka_mids = 0.3 * (self.franka_upper_limits + self.franka_lower_limits)
        self.default_dof_pos = np.zeros(self.franka_num_dofs, dtype=np.float32)
        self.default_dof_pos[:7] = franka_mids[:7]

    def init_default_dof_grippers_open(self):
        # grippers open
        self.default_dof_pos[7:] = self.franka_upper_limits[7:]

    def init_default_dof_state(self):
        self.default_dof_state = np.zeros(self.franka_num_dofs, gymapi.DofState.dtype)
        self.default_dof_state["pos"] = self.default_dof_pos

    def get_franka_dof_props(self):
        return self.franka_dof_props
    
    def get_default_dof_state(self):
        return self.default_dof_state
    
    def get_default_dof_pos(self):
        return self.default_dof_pos



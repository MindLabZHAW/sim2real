class SimulationData:
    def __init__(self, net_cf, panda_idxs, rb_states, box_idxs, hand_idxs, down_dir, controller, dof_pos, corners, num_envs, init_pos, init_rot, down_q, pos_action, effort_action, hand_restart,
                 j_eef, mm, dof_vel, default_dof_pos_tensor, root_state):
        self.net_cf = net_cf
        self.panda_idxs = panda_idxs
        self.rb_states = rb_states
        self.box_idxs = box_idxs
        self.hand_idxs = hand_idxs
        self.down_dir = down_dir
        self.controller = controller
        self.dof_pos = dof_pos
        self.corners = corners
        self.num_envs = num_envs
        self.init_pos = init_pos
        self.init_rot = init_rot
        self.down_q = down_q
        self.pos_action = pos_action
        self.effort_action = effort_action
        self.hand_restart = hand_restart
        self.j_eef = j_eef
        self.mm = mm
        self.dof_vel = dof_vel
        self.default_dof_pos_tensor = default_dof_pos_tensor
        self.root_state = root_state
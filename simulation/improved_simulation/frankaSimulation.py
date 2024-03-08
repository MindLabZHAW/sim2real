"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Franka Cube Pick
----------------
Use Jacobian matrix and inverse kinematics control of Franka robot to pick up a box.
Damped Least Squares method from: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
"""

'''

changed by Maryam Rezayati
conda activate rlgpu
export LD_LIBRARY_PATH=/home/mindlab/miniconda3/envs/rlgpu/lib/
python frankaSimulation.py
'''
import os
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import pickle as pkl
from simulation.IndexDataProcessor import IndexDataProcessor
from utils import utils
from assets.assetFactory import AssetFactory
from simulation.SimulationData import SimulationData
from simulation.simulation import Simulation
from simulation.simulationCreator import SimulationCreator
from config.config import Configuration
from simulation.DofDataProcessor import DofDataProcessor

# set random seed
np.random.seed(42)
gym = gymapi.acquire_gym()

# parse arguments

# Add custom arguments
custom_parameters = [
    {"name": "--controller", "type": str, "default": "ik",
     "help": "Controller to use for Franka. Options are {ik, osc}"},
    {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
]
args = gymutil.parse_arguments(
    description="Franka Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
    custom_parameters=custom_parameters,
)

# Grab controller
controller = args.controller
print(controller)
assert controller in {"ik", "osc"}, f"Invalid controller specified -- options are (ik, osc). Got: {controller}"

# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# configure sim
sim_params = Configuration.configure_sim_params(args)

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

asset_creator = AssetFactory(gym, sim)
franka_asset = asset_creator.create_franka_asset()

# configure franka dofs
dof_data_processor = DofDataProcessor(gym, franka_asset)
dof_data_processor.process_dof_data(controller)

# configure env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
print("Creating %d environments" % num_envs)


simulation_creator = SimulationCreator(gym, asset_creator.create_table_asset(), asset_creator.create_box_asset(), asset_creator.create_barrier_asset(), franka_asset)
simulation_creator.create_simulation(num_envs, AssetFactory.TABLE_DIMS, sim, num_per_row, dof_data_processor.get_franka_dof_props(),
                                   dof_data_processor.get_default_dof_state(), dof_data_processor.get_default_dof_pos())

envs = simulation_creator.get_envs()
box_idxs = simulation_creator.get_box_idxs()
hand_idxs = simulation_creator.get_hand_idxs()
panda_idxs = simulation_creator.get_panda_idxs()
init_pos_list = simulation_creator.get_init_pos_list()
init_rot_list = simulation_creator.get_init_rot_list()

utils.point_camera_at_middle_env(gym, viewer, num_envs, envs, num_per_row)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)

# hand orientation for grasping
down_q = torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((num_envs, 4))

# box corner coords, used to determine grasping yaw
box_half_size = 0.5 * AssetFactory.BOX_SIZE
corner_coord = torch.Tensor([box_half_size, box_half_size, box_half_size])
corners = torch.stack(num_envs * [corner_coord]).to(device)

# downard axis
down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

# get jacobian tensor
# for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "franka")
jacobian = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to franka hand
# get link index of panda hand, which we will use as end effector
franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
num_franka_shapes = gym.get_asset_rigid_shape_count(franka_asset)
print(franka_link_dict)
franka_hand_index = franka_link_dict["panda_hand"]
j_eef = jacobian[:, franka_hand_index - 1, :, :7]

# get mass matrix tensor
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "franka")
mm = gymtorch.wrap_tensor(_massmatrix)
mm = mm[:, :7, :7]          # only need elements corresponding to the franka arm

# get rigid body state tensor__pycache__/**/*
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, 9, 1)
dof_vel = dof_states[:, 1].view(num_envs, 9, 1)

# get net contactforse tensor
_net_cf = gym.acquire_net_contact_force_tensor(sim)
net_cf = gymtorch.wrap_tensor(_net_cf)

# Create a tensor noting whether the hand should return to the initial position
hand_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)

# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)
effort_action = torch.zeros_like(pos_action)
contacted_link =torch.zeros_like(torch.tensor(panda_idxs))

flat_list = []
for sublist in panda_idxs:
    for item in sublist:
        flat_list.append(item)
panda_idxs = flat_list

# send to torch
default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

simulation_data = SimulationData(net_cf, panda_idxs, rb_states, box_idxs, hand_idxs, down_dir, controller, dof_pos, corners, num_envs, init_pos, init_rot, down_q, pos_action,
                                  effort_action, hand_restart, j_eef, Configuration.DAMPING, mm, Configuration.KP, Configuration.KP_NULL, Configuration.KD, Configuration.KD_NULL, 
                                  dof_vel, default_dof_pos_tensor)

simulation = Simulation(gym, sim, viewer, device, simulation_data)
simulation.run(15)

# save data
dataPath = os.getcwd()+'/DATA/data.pickle'
pkl.dump(simulation.get_data_dict(), open(dataPath, 'wb'))

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


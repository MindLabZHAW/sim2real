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
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import pickle as pkl
from simulation.improved_simulation.simulation.TensorDataProcessor import TensorDataProcessor
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

default_dof_pos = dof_data_processor.get_default_dof_pos()

simulation_creator = SimulationCreator(gym, asset_creator.create_table_asset(), asset_creator.create_box_asset(), asset_creator.create_barrier_asset(), franka_asset)
simulation_creator.create_simulation(num_envs, AssetFactory.TABLE_DIMS, sim, num_per_row, dof_data_processor.get_franka_dof_props(),
                                   dof_data_processor.get_default_dof_state(), default_dof_pos)

utils.point_camera_at_middle_env(gym, viewer, num_envs, simulation_creator.get_envs(), num_per_row)


gym.prepare_sim(sim)

panda_idxs = simulation_creator.get_panda_idxs()

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
tensor_data_processor = TensorDataProcessor(num_envs, simulation_creator.get_init_pos_list(), simulation_creator.get_init_rot_list(), panda_idxs)
tensor_data_processor.process_tensor_data(device)

simulation_data = SimulationData(tensor_data_processor.get_net_contact_force_tensor(), utils.get_flat_list(panda_idxs), tensor_data_processor.get_rigib_body_states_tensor(), 
                                 simulation_creator.get_box_idxs(), simulation_creator.get_hand_idxs(), tensor_data_processor.get_down_dir_tensor(), controller, 
                                 tensor_data_processor.get_dof_pos_tensor(), tensor_data_processor.get_box_corners_tensor(), num_envs, tensor_data_processor.get_init_pos_tensor(), 
                                 tensor_data_processor.get_init_rot_tensor(), tensor_data_processor.get_down_q_tensor(), tensor_data_processor.get_pos_action_tensor(),
                                  tensor_data_processor.get_effort_action_tensor(), tensor_data_processor.get_hand_restart_tensor(), 
                                  utils.get_jacabian_end_effector(gym, franka_asset, tensor_data_processor.get_jacobian_tensor()), 
                                  tensor_data_processor.get_mass_matrix_tensor(), tensor_data_processor.get_dof_vel_tensor(), 
                                  tensor_data_processor.get_default_pos_tensor())

simulation = Simulation(gym, sim, viewer, device, simulation_data)
simulation.run(15)

# save data
dataPath = os.getcwd()+'/DATA/data.pickle'
pkl.dump(simulation.get_data_dict(), open(dataPath, 'wb'))

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


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
import time
import pickle as pkl
from assets.assetFactory import AssetFactory

from utils import utils

dataPath = os.getcwd()+'/DATA/data.pickle'
computer_name = '/home/' + os.getcwd().split('/')[2] + '/'
duration_time = 15


# set random seed
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
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
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 1
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# Set controller parameters
# IK params
damping = 0.05

# OSC params
kp = 150.
kd = 2.0 * np.sqrt(kp)
kp_null = 10.
kd_null = 2.0 * np.sqrt(kp_null)

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

asset_creator = AssetFactory(gym, sim)

table_asset = asset_creator.create_table_asset()
box_asset = asset_creator.create_box_asset()
barrier_asset = asset_creator.create_barrier_asset()
franka_asset = asset_creator.create_franka_asset()

# configure franka dofs
franka_dof_props = gym.get_asset_dof_properties(franka_asset)
franka_lower_limits = franka_dof_props["lower"]
franka_upper_limits = franka_dof_props["upper"]
franka_ranges = franka_upper_limits - franka_lower_limits
franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

# use position drive for all dofs
if controller == "ik":
    franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
    franka_dof_props["stiffness"][:7].fill(400.0)
    franka_dof_props["damping"][:7].fill(40.0)
else:       # osc
    franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
    franka_dof_props["stiffness"][:7].fill(0.0)
    franka_dof_props["damping"][:7].fill(0.0)
# grippers
franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][7:].fill(800.0)
franka_dof_props["damping"][7:].fill(40.0)

# default dof states and position targets
franka_num_dofs = gym.get_asset_dof_count(franka_asset)
default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
default_dof_pos[:7] = franka_mids[:7]

# grippers open
default_dof_pos[7:] = franka_upper_limits[7:]

default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# send to torch
default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

# get link index of panda hand, which we will use as end effector
franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
num_franka_shapes = gym.get_asset_rigid_shape_count(franka_asset)
print(franka_link_dict)
franka_hand_index = franka_link_dict["panda_hand"]

# configure env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(0, 0, 0)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * AssetFactory.TABLE_DIMS.z)

box_pose = gymapi.Transform()
barrier_pose = gymapi.Transform()

envs = []
box_idxs = []
hand_idxs = []
panda_idxs = []
init_pos_list = []
init_rot_list = []

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

    # add box
    box_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
    box_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
    box_pose.p.z = AssetFactory.TABLE_DIMS.z + 0.5 * AssetFactory.BOX_SIZE
    box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 1)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # get global index of box in rigid body state tensor
    box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
    box_idxs.append(box_idx)
    
    #add barriers 
    #1
    barrier_pose.p.x = box_pose.p.x + 0.1#np.random.uniform(-0.1, 0.1)
    barrier_pose.p.y = box_pose.p.y - 0.1#np.random.uniform(-0.1, 0.1)
    barrier_pose.p.z = table_pose.p.z + 0.5 * AssetFactory.BARRIER_DIMS.z

    #barrier_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),np.random.uniform(-math.pi, math.pi))
    barrier_handle = gym.create_actor(env, barrier_asset, barrier_pose, "barrier0", i, 4)
    color = gymapi.Vec3(1, 0, 0)
    gym.set_rigid_body_color(env, barrier_handle, 4, gymapi.MESH_VISUAL_AND_COLLISION, color)

    #2
    barrier_pose.p.x = box_pose.p.x - 0.1#np.random.uniform(-0.1, 0.1)
    barrier_pose.p.y = box_pose.p.y + 0.1#np.random.uniform(-0.1, 0.1)
    barrier_pose.p.z = table_pose.p.z + 0.5 * AssetFactory.BARRIER_DIMS.z

    #barrier_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),np.random.uniform(-math.pi, math.pi))
    barrier_handle_1 = gym.create_actor(env, barrier_asset, barrier_pose, "barrier1", i, 6)
    color = gymapi.Vec3(0, 1, np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, barrier_handle_1, 6, gymapi.MESH_VISUAL_AND_COLLISION, color)
    
    #3
    barrier_pose.p.x = box_pose.p.x - 0.2#np.random.uniform(-0.1, 0.1)
    barrier_pose.p.y = box_pose.p.y + 0.2#np.random.uniform(-0.1, 0.1)
    barrier_pose.p.z = table_pose.p.z + 0.5 * AssetFactory.BARRIER_DIMS.z

    #barrier_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),np.random.uniform(-math.pi, math.pi))
    barrier_handle_1 = gym.create_actor(env, barrier_asset, barrier_pose, "barrier1", i, 8)
    color = gymapi.Vec3(0, 1, np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, barrier_handle_1, 8, gymapi.MESH_VISUAL_AND_COLLISION, color)
    # add franka
    franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 2)

    # set dof properties
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

    # get inital hand pose
    hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)
    panda_idxs.append(range(hand_idx-8,hand_idx+1)) #+3


# point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

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
j_eef = jacobian[:, franka_hand_index - 1, :, :7]

# get mass matrix tensor
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "franka")
mm = gymtorch.wrap_tensor(_massmatrix)
mm = mm[:, :7, :7]          # only need elements corresponding to the franka arm

# get rigid body state tensor
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
# simulation loop
start_time = time.time()

data_dict = []
#df_master = pd.DataFrame(())
index_number = 0
while time.time()-start_time < duration_time: #not gym.query_viewer_has_closed(viewer):
    #time.sleep(0.1)
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)
    gym.refresh_net_contact_force_tensor(sim)

    #contact detection
    #print(net_cf)
    #print(panda_idxs)
    #print(net_cf[panda_idxs])
    contacted_link = utils.count_nonzero(net_cf[panda_idxs])
    if torch.count_nonzero(contacted_link) == 0:
        print('There is no Contact :)')
    else:
        print('There is a Contact :(')
        #print(contacted_link)
        print(net_cf[panda_idxs])
    #control
    
    box_pos = rb_states[box_idxs, :3]
    box_rot = rb_states[box_idxs, 3:7]

    hand_pos = rb_states[hand_idxs, :3]
    hand_rot = rb_states[hand_idxs, 3:7]
    hand_vel = rb_states[hand_idxs, 7:]

    to_box = box_pos - hand_pos
    box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
    box_dir = to_box / box_dist
    box_dot = box_dir @ down_dir.view(3, 1)

    # how far the hand should be from box for grasping
    grasp_offset = 0.11 if controller == "ik" else 0.10

    # determine if we're holding the box (grippers are closed and box is near)
    gripper_sep = dof_pos[:, 7] + dof_pos[:, 8]
    gripped = (gripper_sep < 0.045) & (box_dist < grasp_offset + 0.5 * AssetFactory.BOX_SIZE) #true or false

    yaw_q = utils.cube_grasping_yaw(box_rot, corners)
    box_yaw_dir = utils.quat_axis(yaw_q, 0)
    hand_yaw_dir = utils.quat_axis(hand_rot, 0)
    yaw_dot = torch.bmm(box_yaw_dir.view(num_envs, 1, 3), hand_yaw_dir.view(num_envs, 3, 1)).squeeze(-1)

    # determine if we have reached the initial position; if so allow the hand to start moving to the box
    to_init = init_pos - hand_pos
    init_dist = torch.norm(to_init, dim=-1)
    hand_restart = (hand_restart & (init_dist > 0.02)).squeeze(-1)
    return_to_start = (hand_restart | gripped.squeeze(-1)).unsqueeze(-1)

    # if hand is above box, descend to grasp offset
    # otherwise, seek a position above the box
    above_box = ((box_dot >= 0.99) & (yaw_dot >= 0.95) & (box_dist < grasp_offset * 3)).squeeze(-1)
    grasp_pos = box_pos.clone()
    grasp_pos[:, 2] = torch.where(above_box, box_pos[:, 2] + grasp_offset, box_pos[:, 2] + grasp_offset * 2.5)

    # compute goal position and orientation
    goal_pos = torch.where(return_to_start, init_pos, grasp_pos)
    goal_rot = torch.where(return_to_start, init_rot, quat_mul(down_q, quat_conjugate(yaw_q)))

    # compute position and orientation error
    pos_err = goal_pos - hand_pos
    orn_err = utils.orientation_error(goal_rot, hand_rot)
    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

    # Deploy control based on type
    if controller == "ik":
        pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + utils.control_ik(dpose, j_eef, device, damping, num_envs)
    else:       # osc
        effort_action[:, :7] = utils.control_osc(dpose, dpose, mm, j_eef, kp, kp_null, kd, kd_null, hand_vel, dof_vel, default_dof_pos_tensor, dof_pos, device)

    # gripper actions depend on distance between hand and box
    close_gripper = (box_dist < grasp_offset + 0.02) | gripped
    # always open the gripper above a certain height, dropping the box and restarting from the beginning
    hand_restart = hand_restart | (box_pos[:, 2] > 0.6)
    keep_going = torch.logical_not(hand_restart)
    close_gripper = close_gripper & keep_going.unsqueeze(-1)
    grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * num_envs).to(device), torch.Tensor([[0.04, 0.04]] * num_envs).to(device))
    pos_action[:, 7:9] = grip_acts

    # Deploy actions
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))

    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)
    if index_number == 0:
        data_dict = { 'time':  [time.time()-start_time], 'contact' : net_cf[panda_idxs][None,:,:]}
    data_dict['time'] = np.append(data_dict['time'],[time.time()-start_time])
    data_dict['contact'] = torch.cat((data_dict['contact'] , net_cf[panda_idxs][None,:,:]))
    index_number = index_number + 1

# cleanup
pkl.dump(data_dict, open(dataPath, 'wb'))
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


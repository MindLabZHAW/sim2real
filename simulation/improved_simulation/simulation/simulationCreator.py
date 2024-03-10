from isaacgym import gymapi
import numpy as np
import math

from simulation.improved_simulation.assets.assetFactory import AssetFactory
from simulation.improved_simulation.config.config import Configuration

class SimulationCreator:
    def __init__(self, gym, table_asset, box_asset, barrier_asset, franka_asset):
        self.gym = gym
        self.table_asset = table_asset
        self.box_asset = box_asset
        self.barrier_asset = barrier_asset
        self.franka_asset = franka_asset
        self.envs = []
        self.box_idxs = []
        self.hand_idxs = []
        self.init_pos_list = []
        self.init_rot_list = []
        self.panda_idxs = []


    def create_simulation(self, number_of_envs, table_dims, sim, num_per_row, franka_dof_props,
                         default_dof_state, default_dof_pos):
        
        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(0, 0, 0)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

        box_pose = gymapi.Transform()
        barrier_pose = gymapi.Transform()

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(sim, plane_params)


        for i in range(number_of_envs):
            # create env
            env = self.gym.create_env(sim, Configuration.ENV_LOWER, Configuration.ENV_UPPER, num_per_row)
            self.envs.append(env)

            # add table
            table_handle = self.gym.create_actor(env, self.table_asset, table_pose, "table", i, 0)

            # add box
            box_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
            box_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
            box_pose.p.z = AssetFactory.TABLE_DIMS.z + 0.5 * AssetFactory.BOX_SIZE
            box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            box_handle = self.gym.create_actor(env, self.box_asset, box_pose, "box", i, 1)
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            # get global index of box in rigid body state tensor
            box_idx = self.gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
            self.box_idxs.append(box_idx)
            
            #add barriers 
            #1
            barrier_pose.p.x = box_pose.p.x + 0.1#np.random.uniform(-0.1, 0.1)
            barrier_pose.p.y = box_pose.p.y - 0.1#np.random.uniform(-0.1, 0.1)
            barrier_pose.p.z = table_pose.p.z + 0.5 * AssetFactory.BARRIER_DIMS.z

            #barrier_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),np.random.uniform(-math.pi, math.pi))
            barrier_handle = self.gym.create_actor(env, self.barrier_asset, barrier_pose, "barrier0", i, 4)
            color = gymapi.Vec3(1, 0, 0)
            self.gym.set_rigid_body_color(env, barrier_handle, 4, gymapi.MESH_VISUAL_AND_COLLISION, color)

            #2
            barrier_pose.p.x = box_pose.p.x - 0.1#np.random.uniform(-0.1, 0.1)
            barrier_pose.p.y = box_pose.p.y + 0.1#np.random.uniform(-0.1, 0.1)
            barrier_pose.p.z = table_pose.p.z + 0.5 * AssetFactory.BARRIER_DIMS.z

            #barrier_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),np.random.uniform(-math.pi, math.pi))
            barrier_handle_1 = self.gym.create_actor(env, self.barrier_asset, barrier_pose, "barrier1", i, 6)
            color = gymapi.Vec3(0, 1, np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env, barrier_handle_1, 6, gymapi.MESH_VISUAL_AND_COLLISION, color)
            
            #3
            barrier_pose.p.x = box_pose.p.x - 0.2#np.random.uniform(-0.1, 0.1)
            barrier_pose.p.y = box_pose.p.y + 0.2#np.random.uniform(-0.1, 0.1)
            barrier_pose.p.z = table_pose.p.z + 0.5 * AssetFactory.BARRIER_DIMS.z

            #barrier_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),np.random.uniform(-math.pi, math.pi))
            barrier_handle_1 = self.gym.create_actor(env, self.barrier_asset, barrier_pose, "barrier1", i, 8)
            color = gymapi.Vec3(0, 1, np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env, barrier_handle_1, 8, gymapi.MESH_VISUAL_AND_COLLISION, color)
            # add franka
            franka_handle = self.gym.create_actor(env, self.franka_asset, franka_pose, "franka", i, 2)

            # set dof properties
            self.gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

            # set initial dof states
            self.gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            self.gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

            # get inital hand pose
            hand_handle = self.gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
            hand_pose = self.gym.get_rigid_transform(env, hand_handle)
            self.init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            self.init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

            # get global index of hand in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)
            self.panda_idxs.append(range(hand_idx-8,hand_idx+1)) #+3


    def get_envs(self):
        return self.envs

    def get_box_idxs(self):
        return self.box_idxs

    def get_hand_idxs(self):
        return self.hand_idxs

    def get_init_pos_list(self):
        return self.init_pos_list

    def get_init_rot_list(self):
        return self.init_rot_list

    def get_panda_idxs(self):
        return self.panda_idxs
from isaacgym import gymapi
import numpy as np
import math

from assets.assetFactory import AssetFactory
from config.config import Configuration


class SimulationCreator:
    FRANKA_INITIAL_POSITION = gymapi.Vec3(0, 0, 0)

    BARRIER_X_RANGE = [0.13, 0.6]
    BARRIER_Y_RANGE = [0.15, 0.3]
    BARRIER_Z_RANGE = [0.2, 0.5]

    NUM_OF_BARRIER_SETS = 2

    def __init__(self, gym, asset_factory, franka_asset):
        self.gym = gym

        self.franka_asset = franka_asset
        self.barrier_asset = None
        self.create_assets(asset_factory)

        self.envs = []
        self.hand_idxs = []
        self.init_pos_list = []
        self.init_rot_list = []
        self.panda_idxs = []
        self.link_dict = []
        self.dof_dict = []

    def create_assets(self, asset_factory):
        self.barrier_asset = asset_factory.create_barrier_asset()

    def create_simulation(self, number_of_envs, sim, num_per_row, franka_dof_props,
                          default_dof_state, default_dof_pos):

        self.add_ground_plane(sim)

        franka_pose = gymapi.Transform()
        franka_pose.p = self.FRANKA_INITIAL_POSITION

        for i in range(number_of_envs):
            # create env
            env = self.gym.create_env(sim, Configuration.ENV_LOWER, Configuration.ENV_UPPER, num_per_row)
            self.envs.append(env)

            # add barrier sets (2 vertical + 1 horizontal per set)
            for j in range(self.NUM_OF_BARRIER_SETS):
                self.add_randomized_barrier_set(env, i, similar_x=True)

            # add franka
            franka_handle = self.gym.create_actor(env, self.franka_asset, franka_pose, "franka", i, 2)
            self.set_up_franka(env, franka_handle, franka_dof_props, default_dof_state, default_dof_pos)
            self.get_initial_hand_pose(env, franka_handle)

            #Get asset data
            self.link_dict = self.gym.get_asset_rigid_body_dict(self.franka_asset)
            self.dof_dict = self.gym.get_asset_dof_dict(self.franka_asset)

    def add_barrier(self, barrier_pose, env, name, i, collision_filter, color):
        barrier_handle = self.gym.create_actor(env, self.barrier_asset, barrier_pose, name, i, collision_filter)
        self.gym.set_rigid_body_color(env, barrier_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    def add_randomized_barrier_set(self, env, i, similar_x):
        """
        similar_x: True if both vertical barriers same x_offset, otherwise both offsets drawn randomly
        """

        # Create 2 vertical barriers
        # Randomized x-offset
        x_offset_1 = np.random.uniform(self.BARRIER_X_RANGE[0], self.BARRIER_X_RANGE[1])

        if similar_x:
            x_offset_2 = x_offset_1
        else:
            x_offset_2 = np.random.uniform(self.BARRIER_X_RANGE[0], self.BARRIER_X_RANGE[1])

        # Randomized Y-offsets for both vertical bars
        y_offset_1 = np.random.uniform(self.BARRIER_Y_RANGE[0], self.BARRIER_Y_RANGE[1])
        y_offset_2 = np.random.uniform(self.BARRIER_Y_RANGE[0], self.BARRIER_Y_RANGE[1]) * (-1)

        # Generate vertical barrier positions
        vertical_barrier_pose1 = self.create_vertical_barrier_pose(x_offset_1, y_offset_1)
        vertical_barrier_pose2 = self.create_vertical_barrier_pose(x_offset_2, y_offset_2)

        # Add barriers to env
        color = gymapi.Vec3(0, 1, np.random.uniform(0, 1))
        self.add_barrier(vertical_barrier_pose1, env, "vertical_barrier1", i, 4, color=color)
        self.add_barrier(vertical_barrier_pose2, env, "vertical_barrier2", i, 4, color=color)

        # Create Horizontal barrier
        # Generate 2 z-offsets through which horizontal barrier goes
        z_offset_1 = np.random.uniform(self.BARRIER_Z_RANGE[0], self.BARRIER_Z_RANGE[1])
        z_offset_2 = np.random.uniform(self.BARRIER_Z_RANGE[0], self.BARRIER_Z_RANGE[1])
        horizontal_barrier_pose = self.generate_rotated_horizontal_barrier_pose(vertical_barrier_pose1,
                                                                                vertical_barrier_pose2,
                                                                                z_offset_1, z_offset_2,
                                                                                is_attached=True)

        self.add_barrier(horizontal_barrier_pose, env, "horizontal_barrier1", i, 4, color=color)

    def create_vertical_barrier_pose(self, x_offset, y_offset):
        barrier_pose = gymapi.Transform()
        barrier_pose.p.x = self.FRANKA_INITIAL_POSITION.x + x_offset
        barrier_pose.p.y = self.FRANKA_INITIAL_POSITION.y + y_offset
        barrier_pose.p.z = AssetFactory.BARRIER_DIMS.z * 0.5 + self.FRANKA_INITIAL_POSITION.z
        print(barrier_pose.p.z)
        return barrier_pose

    def generate_rotated_horizontal_barrier_pose(self, barrier_pose1, barrier_pose2, z_offset1, z_offset2, is_attached):
        """
        is_attached: boolean -> True if horizontal barrier attached to vertical barriers, False if goes "through" vertical barriers
        """
        x_offset = AssetFactory.BARRIER_DIMS.x if is_attached else 0

        point1 = np.array([barrier_pose1.p.x - x_offset, barrier_pose1.p.y, z_offset1])
        point2 = np.array([barrier_pose2.p.x - x_offset, barrier_pose2.p.y, z_offset2])

        rotation_quaternion = self.calculate_rotation_quaternion_for_direction_between_two_points(point1, point2)

        barrier_pose = gymapi.Transform()
        barrier_pose.p.x = (point1[0] + point2[0]) / 2
        barrier_pose.p.y = (point1[1] + point2[1]) / 2
        barrier_pose.p.z = (point1[2] + point2[2]) / 2
        barrier_pose.r = rotation_quaternion
        return barrier_pose

    def calculate_rotation_quaternion_for_direction_between_two_points(self, point1, point2):
        # Calculate the direction vector between the two points
        direction_vector = point2 - point1
        direction_vector /= np.linalg.norm(direction_vector)  # Normalize the direction vector

        # Define the object's initial forward vector
        object_forward_vector = np.array([0.0, 0.0, 1.0])

        # Calculate the axis of rotation using cross product
        rotation_axis = np.cross(object_forward_vector, direction_vector)
        rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize the rotation axis

        # Calculate the angle of rotation using dot product
        dot_product = np.dot(object_forward_vector, direction_vector)
        rotation_angle = np.arccos(dot_product)

        # Construct the quaternion for rotation
        rotation_quaternion = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(rotation_axis[0], rotation_axis[1], rotation_axis[2]), rotation_angle)
        return rotation_quaternion

    def set_up_franka(self, env, franka_handle, franka_dof_props, default_dof_state, default_dof_pos):
        # set dof properties
        self.gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)
        # set initial dof states
        self.gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)
        # set initial position targets
        self.gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

    def add_ground_plane(self, sim):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(sim, plane_params)

    def add_object(self, env, asset, pose, name, env_index, collision_filter):
        handle = self.gym.create_actor(env, asset, pose, name, env_index, collision_filter)
        return handle

    def get_initial_hand_pose(self, env, franka_handle):
        # get inital hand pose
        hand_handle = self.gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
        hand_pose = self.gym.get_rigid_transform(env, hand_handle)
        self.init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
        self.init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

        # get global index of hand in rigid body state tensor
        hand_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
        self.hand_idxs.append(hand_idx)
        self.panda_idxs.append(range(hand_idx - 8, hand_idx + 1))  # +3

    def get_envs(self):
        return self.envs

    def get_hand_idxs(self):
        return self.hand_idxs

    def get_init_pos_list(self):
        return self.init_pos_list

    def get_init_rot_list(self):
        return self.init_rot_list

    def get_panda_idxs(self):
        return self.panda_idxs
    
    def get_asset_data(self):
        return self.link_dict, self.dof_dict
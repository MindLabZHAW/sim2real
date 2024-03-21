from isaacgym import gymapi
import os

class AssetFactory:
    BARRIER_DIMS = gymapi.Vec3(0.05, 0.05, 1)

    def __init__(self, gym, sim):
        self.gym = gym
        self.sim = sim
    
    def create_barrier_asset(self):
        # create barrier asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        return self.gym.create_box(self.sim, self.BARRIER_DIMS.x, self.BARRIER_DIMS.y, self.BARRIER_DIMS.z, asset_options)

    def create_franka_asset(self):
        # load franka asset
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.flip_visual_attachments = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        computer_name = '/home/' + os.getcwd().split('/')[2] + '/'
        asset_root = computer_name + "/isaacgym/assets"
        return self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)






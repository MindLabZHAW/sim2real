
from isaacgym import gymapi
import numpy as np

class Configuration:
    # Set controller parameters
    # IK params
    DAMPING = 0.05

    # OSC params
    KP = 150.
    KD = 2.0 * np.sqrt(KP)
    KP_NULL = 10.
    KD_NULL = 2.0 * np.sqrt(KP_NULL)

    @staticmethod
    def configure_sim_params(args):
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
        return sim_params
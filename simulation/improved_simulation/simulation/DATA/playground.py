import pickle
import torch
import io
import numpy as np

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=torch.device('cpu'))
        else: 
            return super().find_class(module, name)

class PlayGround:
    def __init__(self, path):
        self.file_path = path
        self.data = self.open_pickle_file()
        
    def open_pickle_file(self):
        with open(self.file_path, 'rb') as f:
            unpickler = CustomUnpickler(f)
            data = unpickler.load()
            return data
    
    def print_data_info(self):
        self.data = self.open_pickle_file()
        print(self.data.keys())
        keys = self.data.keys()
        for key in keys:
            print(key)
            if key == "joint_velocity1":
                print("Length for joint velocity list:", len(self.data[key]))
            else:
                print("Shape for:", key, self.data[key].shape)

def extract_velocities(jacobian_tensor):
    """
    Extracts angular, linear, and joint velocities from a Jacobian tensor.

    Parameters:
        jacobian_tensor (torch.Tensor): Jacobian tensor with shape [num_entries(=time entries), num_envs(amouont of environments defined in the simulation), 11 (all links), 6, 9 (all joints)].

    Returns:
        angular_velocity (torch.Tensor): Angular velocity tensor with shape [num_entries(=time entries), num_envs(amouont of environments defined in the simulation), 11 (all links), 3 (linear velocities), 9 (all joints)].
        linear_velocity (torch.Tensor): Linear velocity tensor with shape [num_entries(=time entries), num_envs(amouont of environments defined in the simulation), 11 (all links), 3 (angular velocities), 9 (all joints)].
        joint_velocity (torch.Tensor): Joint velocity tensor with shape [num_entries(=time entries), num_envs(amouont of environments defined in the simulation), 11 (all links), 6, 9 (all joints)].
    """
    num_entries = jacobian_tensor.shape[0]
    num_envs = jacobian_tensor.shape[1]

    print("Jacobian Shape:", jacobian_tensor.shape)

    # Extract Angular Velocity
    angular_velocity = jacobian_tensor[:, :, :, -3:, :]  # Slicing for the last axis
    print("Angular Velocity Shape (before reshape):", angular_velocity.shape)

    # Extract Linear Velocity
    linear_velocity = jacobian_tensor[:, :, :, :3, :]  # Slicing for the last axis
    print("Linear Velocity Shape (before reshape):", linear_velocity.shape)

    # Extract Joint Velocity
    joint_velocity = jacobian_tensor[:, :, :, :6, :]  # Slicing for the last axis
    print("Joint Velocity Shape (before reshape):", joint_velocity.shape)


    return angular_velocity, linear_velocity, joint_velocity



playground_jacobian = PlayGround('./jacobian.pickle')   
print("jacobian")
jacobian_data = playground_jacobian.print_data_info() 
angular_velocity, linear_velocity, joint_velocity = extract_velocities(playground_jacobian.data['jacobian'])
print("angular_velocity")
print(angular_velocity[0])
print(angular_velocity.shape)
print("linear_velocity")
print(linear_velocity[0])
print(linear_velocity.shape)
print("joint_velocity")
print(joint_velocity[0, 0, 0, 0, :])
print(joint_velocity.shape)

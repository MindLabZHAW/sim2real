import random
import numpy as np

from simulation.movement.Movement import Movement

class ForcedContactMovement(Movement):
    X_OFFSET = [-0.2, 0.2]
    Y_OFFSET = [-0.2, 0.2]
    Z_OFFSET = [-0.2, 0.2]

    def __init__(self, initial_position, initial_rotation):
         self.count_movement = 0
         self.barrier_idx = 0
         super().__init__(initial_position, initial_rotation)

    def get_next_position(self, rigid_body_state_tensor):
        self.count_movement += 1
        if(self.count_movement % 2 == 0):
            return self.initial_position
        else:
            return self.get_next_position_tensor(rigid_body_state_tensor)
        
    def get_next_rotation(self):
        return self.current_rotation
        
    
    def get_next_position_tensor(self, rigid_body_state_tensor):
            barrier_position_tensor = rigid_body_state_tensor[self.barrier_idx % 6, :3]
            self.barrier_idx += 1
            return self.add_offset_to(barrier_position_tensor)
    
    def add_offset_to(self, tensor):
        tensor_temp = tensor.clone()
        tensor_temp[0] += random.uniform(self.X_OFFSET[0], self.X_OFFSET[1])
        tensor_temp[1] += random.uniform(self.Y_OFFSET[0], self.Y_OFFSET[1])
        tensor_temp[2] += random.uniform(self.Z_OFFSET[0], self.Z_OFFSET[1])
        return tensor_temp
        
import random
import numpy as np
import numpy as np

class ForcedContactMovement():
    def __init__(self, initial_position):
        self.initial_position = initial_position
        self.current_position = initial_position
        self.count_movement = 0
        super().__init__()
    

    def get_coordinates_for_next_position(self, rigid_body_state_tensor, index_number):
        
        if index_number % 50 == 0:
            self.current_position = self.get_next_position(rigid_body_state_tensor)
        
        return self.current_position
    
    def get_next_position(self, rigid_body_state_tensor):
        self.count_movement += 1
        if(self.count_movement % 2 == 0):
            return self.initial_position
        else:
            rand_barrier_idx = self.get_rand_barrier_index()
            barrier_position_tensor = rigid_body_state_tensor[rand_barrier_idx, :3]
            return self.add_offset_to(barrier_position_tensor)


    def add_offset_to(self, tensor):
        tensor[0] += random.uniform(-0.3, 0.3)
        tensor[1] += random.uniform(-0.3, 0.3)
        tensor[2] += random.uniform(-0.3, 0.3)
        return tensor
        

    def get_rand_barrier_index(self):
        return random.randint(2,8)
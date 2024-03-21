import random
import numpy as np

from simulation.movement.Movement import Movement

class ForcedContactMovement(Movement):
    OFFSET = 0.3

    def __init__(self, initial_position):
         super().__init__(initial_position)
        
    
    def get_next_position_tensor(self, rigid_body_state_tensor):
            rand_barrier_idx = self.get_rand_barrier_index()
            barrier_position_tensor = rigid_body_state_tensor[rand_barrier_idx, :3]
            return super().add_offset_to(barrier_position_tensor, self.OFFSET)
        

    def get_rand_barrier_index(self):
        return random.randint(2,8)
        
import random
from simulation.movement.Movement import Movement


class RandomMovement(Movement):
    X_OFFSET = [-0.05, 0.15]
    Y_OFFSET = [-0.3, 0.3]
    Z_OFFSET =  [-0.1, 0.1]

    def __init__(self, initial_position, initial_rotation):
        self.count_movement = 0
        super().__init__(initial_position, initial_rotation)

    def get_next_position(self, rigid_body_state_tensor):
        self.count_movement += 1
        if self.count_movement % 10 == 0:
            return self.initial_position
        else:
            current_pos = self.current_position.clone()
            current_pos[0][0] += random.uniform(self.X_OFFSET[0], self.X_OFFSET[1])
            current_pos[0][1] += random.uniform(self.Y_OFFSET[0], self.Y_OFFSET[1])
            current_pos[0][2] += random.uniform(self.Z_OFFSET[0], self.Z_OFFSET[1])
            return current_pos
    
    def get_next_rotation(self):
        return self.initial_rotation
        
    
    
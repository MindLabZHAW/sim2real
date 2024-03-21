from simulation.movement.Movement import Movement


class RectangleMovement(Movement):

    def __init__(self, initial_position):
        super().__init__(initial_position)


    def get_next_position(self, rigid_body_state_tensor):
        

        if self.count_movement % 4 == 0:
            # go a bit right
            self.current_position[0][1] -= 0.15
        
        return self.current_position
        
 
    def get_next_rotation(self):
        return self.initial_rotation

    
    
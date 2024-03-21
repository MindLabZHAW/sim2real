from simulation.movement.Movement import Movement


class RandomMovement(Movement):
    RAND_OFFSET = 0.75

    def __init__(self, initial_position):
        super().__init__(initial_position)
    
    def get_next_position_tensor(self, rigid_body_state_tensor):
        return super().add_offset_to(self.initial_position.clone()[0], self.RAND_OFFSET)
    
    
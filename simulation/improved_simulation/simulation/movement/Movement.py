from abc import ABC, abstractmethod
import random

class Movement:
    INITAL_POS_Z_OFFSET = -0.2
    MOVEMENT_AFTER_AMOUNT_OF_STEPS = [20, 40]

    def __init__(self, initial_position, initial_rotation) -> None:
        self.initial_position = initial_position
        self.initial_position[0][2] += self.INITAL_POS_Z_OFFSET

        self.current_position = initial_position
        self.initial_rotation = initial_rotation
        self.current_rotation = initial_rotation
        self.step_count = 0
        self.do_next_position_count = random.randint(self.MOVEMENT_AFTER_AMOUNT_OF_STEPS[0] ,self.MOVEMENT_AFTER_AMOUNT_OF_STEPS[1])


    def get_coordinates_and_rotation_for_next_position(self, rigid_body_state_tensor, index_number):
        if self.step_count == self.do_next_position_count:
            self.current_position = self.get_next_position(rigid_body_state_tensor)
            self.current_rotation = self.get_next_rotation()
            self.step_count = 0
            self.do_next_position_count = random.randint(self.MOVEMENT_AFTER_AMOUNT_OF_STEPS[0] ,self.MOVEMENT_AFTER_AMOUNT_OF_STEPS[1])

        self.step_count += 1
        return self.current_position, self.current_rotation
    

    @abstractmethod
    def get_next_position(self, rigid_body_state_tensor):
        pass

    @abstractmethod
    def get_next_rotation(self):
        pass

    def add_offset_to(self, tensor, offset):
        tensor[0] += random.uniform(-offset, offset)
        tensor[1] += random.uniform(-offset, offset)
        tensor[2] += random.uniform(-offset, offset)
        return tensor

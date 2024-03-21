from abc import ABC, abstractmethod
import random

class Movement:

    def __init__(self, initial_position) -> None:
        self.initial_position = initial_position
        self.current_position = initial_position
        self.count_movement = 0


    def get_coordinates_for_next_position(self, rigid_body_state_tensor, index_number):

        if index_number % 50 == 0:
            self.current_position = self.get_next_position(rigid_body_state_tensor)
        
        return self.current_position
    

    def get_next_position(self, rigid_body_state_tensor):
        self.count_movement += 1
        if(self.count_movement % 2 == 0):
            return self.initial_position
        else:
            return self.get_next_position_tensor(rigid_body_state_tensor)
        

    @abstractmethod
    def get_next_position_tensor(self, rigid_body_state_tensor):
        pass


    def add_offset_to(self, tensor, offset):
        tensor[0] += random.uniform(-offset, offset)
        tensor[1] += random.uniform(-offset, offset)
        tensor[2] += random.uniform(-offset, offset)
        return tensor

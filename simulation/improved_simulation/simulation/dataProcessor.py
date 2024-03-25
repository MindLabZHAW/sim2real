import os
import torch
from utils import utils
import numpy as np
import time
import pickle as pkl

class DataProcessor:
    def __init__(self, data, start_time, gym, sim):
        self.sim_data = data
        self.start_time = start_time
        self.cf_data_dict = []
        self.rb_state_data_dict = []
        self.root_state_data_dict = []
        self.jacobian_data_dict = []
        self.index_number = 0
        self.gym = gym
        self.sim = sim
        self.joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", 
                            "panda_joint5", "panda_joint6", "panda_joint7", 
                            "panda_finger_joint1", "panda_finger_joint2"]
        self.joint_velocities_tensor = None
    
    def cleanup(self):
        if os.path.exists(os.getcwd()+'/simulation/DATA/contact_force_data.pickle'):
            os.remove(os.getcwd()+'/simulation/DATA/contact_force_data.pickle')
            print("File deleted")
        if os.path.exists(os.getcwd()+'/simulation/DATA/rb_state_data.pickle'):
            os.remove(os.getcwd()+'/simulation/DATA/rb_state_data.pickle')
        if os.path.exists(os.getcwd()+'/simulation/DATA/dof_state_data.pickle'):
            os.remove(os.getcwd()+'/simulation/DATA/dof_state_data.pickle')
        if os.path.exists(os.getcwd()+'/simulation/DATA/root_state_data.pickle'):
            os.remove(os.getcwd()+'/simulation/DATA/root_state_data.pickle')
        if os.path.exists(os.getcwd()+'/simulation/DATA/jacobian.pickle'):
            os.remove(os.getcwd()+'/simulation/DATA/jacobian.pickle')

    def process(self):
        #Process the data here
        self.process_contact_force_data()
        #self.process_rb_state_data()
        self.process_jacobian_data()
        #self.process_dof_state_data()
        #self.process_root_state_data()
        self.index_number += 1
        pass
    
    def save_data(self):
        #save the data into the respective pickle files
        self.save_contact_force_data()
        #self.save_rb_state_data()
        self.save_jacobian_data()
        #self.save_dof_state_data()
        #self.save_root_state_data()
        pass
    
    def process_contact_force_data(self):
        if self.index_number == 0:
            self.cf_data_dict = { 'time':  [time.time()-self.start_time], 'contact' : self.sim_data.net_cf[self.sim_data.panda_idxs][None,:,:]}
        self.cf_data_dict['time'] = np.append(self.cf_data_dict['time'],[time.time()-self.start_time])
        self.cf_data_dict['contact'] = torch.cat((self.cf_data_dict['contact'] , self.sim_data.net_cf[self.sim_data.panda_idxs][None,:,:]))

    def process_joint_data(self):
        env = self.gym.get_env(self.sim, 0)
        joint_velocities_list = []
        for joint_name in self.joint_names:
            joint_handle = self.gym.get_joint_handle(env, "franka", joint_name)
            joint_velocities = self.gym.get_joint_velocity(env, joint_handle)
            joint_velocities_list.append(joint_velocities)
        self.joint_velocities_tensor = torch.tensor(joint_velocities_list)
        
        return self.joint_velocities_tensor
        
    def process_rb_state_data(self):
        rb_state = self.sim_data.rb_states
        if self.index_number == 0:
            self.joint_velocities_tensor = self.process_joint_data().unsqueeze(0)  # Initialize the tensor with shape (1, num_joints)
            self.rb_state_data_dict = {'time': torch.tensor([time.time()-self.start_time]), 
                                       'rb_state_position': torch.tensor(rb_state[self.sim_data.panda_idxs][0:3]), 
                                       'rb_state_rotation': torch.tensor(rb_state[self.sim_data.panda_idxs][3:7]), 
                                       'rb_state_velocity': torch.tensor(rb_state[self.sim_data.panda_idxs][7:13]),
                                       'rb_state_linear_velocity': torch.tensor(rb_state[self.sim_data.panda_idxs][7:10]), 
                                       'rb_state_angular_velocity': torch.tensor(rb_state[self.sim_data.panda_idxs][10:13]), 
                                       "joint_velocities": self.joint_velocities_tensor}

        self.rb_state_data_dict['time'] = torch.cat((self.rb_state_data_dict['time'], torch.tensor([time.time()-self.start_time])))
        self.rb_state_data_dict['rb_state_position'] = torch.cat((self.rb_state_data_dict['rb_state_position'], torch.tensor(rb_state[self.sim_data.panda_idxs][0:3])))
        self.rb_state_data_dict['rb_state_rotation'] = torch.cat((self.rb_state_data_dict['rb_state_rotation'], torch.tensor(rb_state[self.sim_data.panda_idxs][3:7])))
        self.rb_state_data_dict['rb_state_velocity'] = torch.cat((self.rb_state_data_dict['rb_state_velocity'], torch.tensor(rb_state[self.sim_data.panda_idxs][7:13])))
        self.rb_state_data_dict['rb_state_linear_velocity'] = torch.cat((self.rb_state_data_dict['rb_state_linear_velocity'], torch.tensor(rb_state[self.sim_data.panda_idxs][7:10])))
        self.rb_state_data_dict['rb_state_angular_velocity'] = torch.cat((self.rb_state_data_dict['rb_state_angular_velocity'], torch.tensor(rb_state[self.sim_data.panda_idxs][10:13])))
        self.joint_velocities_tensor = torch.cat((self.joint_velocities_tensor, self.process_joint_data().unsqueeze(0)), dim=0)
        self.rb_state_data_dict['joint_velocities'] = self.joint_velocities_tensor
    
    def process_jacobian_data(self):
        if self.index_number == 0:
            self.jacobian_data_dict = {'time': [time.time()-self.start_time], 'jacobian': self.sim_data.jacobian[None,:,:]}
        self.jacobian_data_dict['time'] = np.append(self.jacobian_data_dict['time'],[time.time()-self.start_time])
        self.jacobian_data_dict['jacobian'] = torch.cat((self.jacobian_data_dict['jacobian'], self.sim_data.jacobian[None,:,:]))

        #link dict
        #{'panda_hand': 8, 'panda_leftfinger': 9, 'panda_link0': 0, 'panda_link1': 1, 'panda_link2': 2, 'panda_link3': 3, 'panda_link4': 4, 'panda_link5': 5, 'panda_link6': 6, 'panda_link7': 7, 'panda_rightfinger': 10}
        #dof dict:
        #{'panda_finger_joint1': 7, 'panda_finger_joint2': 8, 'panda_joint1': 0, 'panda_joint2': 1, 'panda_joint3': 2, 'panda_joint4': 3, 'panda_joint5': 4, 'panda_joint6': 5, 'panda_joint7': 6}
        pass
    
    def process_root_state_data(self):
        #process the root_state data 
        root_state = self.sim_data.root_state
        #print("root_state:",root_state)
        #print("root_state:",root_state.shape)
        if self.index_number == 0:
            self.root_state_data_dict = {'time': [time.time()-self.start_time], 'root_state_position': root_state[:][0:3], 'root_state_rotation': root_state[:][3:7], 'root_state_velocity': root_state[:][7:]}
        self.root_state_data_dict['time'] = np.append(self.root_state_data_dict['time'],[time.time()-self.start_time])
        self.root_state_data_dict['root_state_position'] = torch.cat((self.root_state_data_dict['root_state_position'], root_state[:][0:3]))
        self.root_state_data_dict['root_state_rotation'] = torch.cat((self.root_state_data_dict['root_state_rotation'], root_state[:][3:7]))
        self.root_state_data_dict['root_state_velocity'] = torch.cat((self.root_state_data_dict['root_state_velocity'], root_state[:][7:]))
    
    def process_dof_state_data(self):
        #process the dof_state data here
        dof_state = self.sim_data.dof_pos
        if self.index_number == 0:
            self.dof_state_data_dict = {'time': [time.time()-self.start_time], 'dof_state_position': dof_state[:, 0], 'dof_state_velocity': dof_state[:, 1]}
        self.dof_state_data_dict['time'] = np.append(self.dof_state_data_dict['time'],[time.time()-self.start_time])
        self.dof_state_data_dict['dof_state_position'] = torch.cat((self.dof_state_data_dict['dof_state_position'], dof_state[:, 0]))
        self.dof_state_data_dict['dof_state_velocity'] = torch.cat((self.dof_state_data_dict['dof_state_velocity'], dof_state[:, 1]))
        pass
    
    def save_contact_force_data(self):
        #save the contact force data into a pickle file
        dataPath = os.getcwd()+'/simulation/DATA/contact_force_data.pickle'
        pkl.dump(self.cf_data_dict, open(dataPath, 'wb'))
        pass
        
    def save_rb_state_data(self):
        #save the rb_state data into a pickle file
        dataPath = os.getcwd()+'/simulation/DATA/rb_state_data.pickle'
        pkl.dump(self.rb_state_data_dict, open(dataPath, 'wb'))
        pass
    
    def save_jacobian_data(self):
        dataPath = os.getcwd()+'/simulation/DATA/jacobian.pickle'
        pkl.dump(self.jacobian_data_dict, open(dataPath, 'wb'))
        pass
    
    def save_root_state_data(self):
        #save the rb_state data into a pickle file
        dataPath = os.getcwd()+'/simulation/DATA/root_state_data.pickle'
        pkl.dump(self.root_state_data_dict, open(dataPath, 'wb'))
        pass
    
    def save_dof_state_data(self):
        #save the dof_state data into a pickle file
        dataPath = os.getcwd()+'/simulation/DATA/dof_state_data.pickle'
        pkl.dump(self.dof_state_data_dict, open(dataPath, 'wb'))
        pass

    def calculate_collision_free_joint_angles(self):
        pass

    def calculate_collision_free_joint_velocities(self):
        pass

    def calculate_collision_free_link_poses(self):
        pass

    def calculate_collision_free_link_velocities(self):
        pass

    def calculate_target_link_poses(self):
        pass
    
    def contact_detection(self):
        # Check whether a contact happened so that the right rb_state data can be saved
        contacted_link = utils.count_nonzero(self.sim_data.net_cf[self.sim_data.panda_idxs])
        if torch.count_nonzero(contacted_link) == 0:
            #print('There is no Contact :)')
            return True
        else:
            #print('There is a Contact :(')
            #print(contacted_link)
            #print(self.sim_data.net_cf[self.sim_data.panda_idxs])
            return False

    def set_label(self):
        #Set the label that a contact has happened: 1=contact / 0=no contact
        if self.contact_detection():
            pass
        else:
            pass
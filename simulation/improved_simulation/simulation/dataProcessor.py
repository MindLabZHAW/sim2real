import os
import torch
from utils import utils
import numpy as np
import time
import pickle as pkl

class DataProcessor:
    def __init__(self, data, start_time):
        self.sim_data = data
        self.start_time = start_time
        self.cf_data_dict = []
        self.rb_state_data_dict = []
        self.index_number = 0
    
    def cleanup(self):
        if os.path.exists(os.getcwd()+'/simulation/DATA/contact_force_data.pickle'):
            os.remove(os.getcwd()+'/simulation/DATA/contact_force_data.pickle')
            print("File deleted")
        if os.path.exists(os.getcwd()+'/simulation/DATA/rb_state_data.pickle'):
            os.remove(os.getcwd()+'/simulation/DATA/rb_state_data.pickle')
        if os.path.exists(os.getcwd()+'/simulation/DATA/dof_state_data.pickle'):
            os.remove(os.getcwd()+'/simulation/DATA/dof_state_data.pickle')

    def process(self):
        #Process the data here
        self.process_contact_force_data()
        self.process_rb_state_data()
        self.process_dof_state_data()
        self.index_number += 1
        pass
    
    def save_data(self):
        #save the data into the respective pickle files
        self.save_contact_force_data()
        self.save_rb_state_data()
        self.save_dof_state_data()
        pass
    
    def process_contact_force_data(self):
        if self.index_number == 0:
            self.cf_data_dict = { 'time':  [time.time()-self.start_time], 'contact' : self.sim_data.net_cf[self.sim_data.panda_idxs][None,:,:]}
        self.cf_data_dict['time'] = np.append(self.cf_data_dict['time'],[time.time()-self.start_time])
        self.cf_data_dict['contact'] = torch.cat((self.cf_data_dict['contact'] , self.sim_data.net_cf[self.sim_data.panda_idxs][None,:,:]))
        
    def process_rb_state_data(self):
        #TODO: Find out why in cf data there is [None,:,:]
        #TODO: Set an additional field for contact label
        #TODO: Test whether the Tensor processing is correct!!!!
        #process the rb_state data here
        rb_state = self.sim_data.rb_states
        if self.index_number == 0:
            self.rb_state_data_dict = {'time': [time.time()-self.start_time], 'rb_state_position': rb_state[self.sim_data.panda_idxs][0:3], 'rb_state_rotation': rb_state[self.sim_data.panda_idxs][3:7], 'rb_state_velocity': rb_state[self.sim_data.panda_idxs][7:]}
        self.rb_state_data_dict['time'] = np.append(self.rb_state_data_dict['time'],[time.time()-self.start_time])
        self.rb_state_data_dict['rb_state_position'] = torch.cat((self.rb_state_data_dict['rb_state_position'], rb_state[self.sim_data.panda_idxs][0:3]))
        self.rb_state_data_dict['rb_state_rotation'] = torch.cat((self.rb_state_data_dict['rb_state_rotation'], rb_state[self.sim_data.panda_idxs][3:7]))
        self.rb_state_data_dict['rb_state_velocity'] = torch.cat((self.rb_state_data_dict['rb_state_velocity'], rb_state[self.sim_data.panda_idxs][7:]))
        #Linear velocity is [7:10] and angular velocity is [10:13]
        pass
    
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
    
    def save_dof_state_data(self):
        #save the dof_state data into a pickle file
        dataPath = os.getcwd()+'/simulation/DATA/dof_state_data.pickle'
        pkl.dump(self.dof_state_data_dict, open(dataPath, 'wb'))
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
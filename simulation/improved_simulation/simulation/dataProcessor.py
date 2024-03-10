import os
import torch
from utils import utils
import numpy as np
import time
import pickle as pkl

class DataProcessor:
    def __init__(self, data, start_time, index_number):
        self.data = data
        self.start_time = start_time
        self.index_number = index_number

    def process(self):
        #Process the data here
        self.process_contact_force_data()
        pass
    
    def save_data(self):
        #save the data into the respective pickle files
        self.save_contact_force_data()
        #TODO: save the rigid body state data
        pass
    
    def process_contact_force_data(self):
        if self.index_number == 0:
            self.cf_data_dict = { 'time':  [time.time()-self.start_time], 'contact' : self.sim_data.net_cf[self.sim_data.panda_idxs][None,:,:]}
        self.cf_data_dict['time'] = np.append(self.cf_data_dict['time'],[time.time()-self.start_time])
        self.cf_data_dict['contact'] = torch.cat((self.cf_data_dict['contact'] , self.sim_data.net_cf[self.sim_data.panda_idxs][None,:,:]))
        
    def save_contact_force_data(self):
        #save the contact force data into a pickle file
        dataPath = os.getcwd()+'/DATA/contact_force_data.pickle'
        pkl.dump(self.cf_data_dict, open(dataPath, 'wb'))
    
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

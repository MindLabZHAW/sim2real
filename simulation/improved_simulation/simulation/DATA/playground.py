import pickle
import torch
import io

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=torch.device('cpu'))
        else: 
            return super().find_class(module, name)

class PlayGround:
    def __init__(self, path):
        self.file_path = path
        self.data = self.open_pickle_file()
        
    def open_pickle_file(self):
        with open(self.file_path, 'rb') as f:
            unpickler = CustomUnpickler(f)
            data = unpickler.load()
            return data
    
    def print_data_info(self):
        self.data = self.open_pickle_file()
        print(self.data.keys())
        keys = self.data.keys()
        for key in keys:
            print(key)
            if key == "joint_velocity1":
                print("Lenght for joint velocity list:",len(self.data[key]))
            else:
                #print("Data for:",key, self.data[key])
                print("Shape for:",key, self.data[key].shape)
                print("[0] for ",key, self.data[key][0])


playground_jacobian = PlayGround('./jacobian.pickle')   
print("jacobian")
jacobian_data = playground_jacobian.print_data_info()     
playground_cf = PlayGround('./contact_force_data.pickle') 
print("cf_data")
cf_data = playground_cf.print_data_info()
print("rb_state_data")
playground_rb = PlayGround('./rb_state_data.pickle')
rb_data = playground_rb.print_data_info()
#print("dof_state_data")
#playground_dof = PlayGround('./dof_state_data.pickle')
#dof_data = playground_dof.print_data_info()
#print("root_state_data")
#playground_root = PlayGround('./root_state_data.pickle')
#root_data = playground_root.print_data_info()

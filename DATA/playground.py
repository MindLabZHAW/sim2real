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
            print("Data for:",key, self.data[key])
            print("Shape for:",key, self.data[key].shape)
            print("[0] for ",key, self.data[key][0])
            
playground = PlayGround('./data.pickle') 
cf_data = playground.print_data_info()




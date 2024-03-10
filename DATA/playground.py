import pickle
import torch
import io

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=torch.device('cpu'))
        else: 
            return super().find_class(module, name)

# Open the pickle file and load the data
with open('./data.pickle', 'rb') as f:
    unpickler = CustomUnpickler(f)
    data = unpickler.load()

# Now you can analyze the data
print(data.keys())
print("time data:", data['time'])
print("time data shape:", data['time'].shape)
print("time[0]:",data['time'][0])
print("contact data:",data['contact'])
print("contact data shape:", data['contact'].shape)
print("contact[]:",data['contact'][0])
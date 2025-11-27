# this is for fetching all the samples ids and then their data from the database, and finally training the neural network model and storing its cache in app/model_cache/model.pth
# the network is the spectrum classifier defined in spectrum_model.py




'''
here is the code of the model
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrumClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SpectrumClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        # the pooling layer makes the input smaller and reduces overfitting, and also reduces the computational load
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 50, 128)  # Adjust input size based on your data
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Define the forward pass
        # Assuming input x has shape (batch_size, 1, 100)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



'''
import sqlite3
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from app.models.spectrum_model import SpectrumClassifier
import os
#import utils
import app.utils as utils
# Define the path to your database
db_file_path = 'database/microplastics_reference.db'


sample_ids=utils.get_all_ids("app/database/microplastics_reference.db")

data_list=[]

label_list=[]
wavenumber_lens=set()
min_wavenumber_sample=set()
max_wavenumber_sample=set()
print(f"Total samples found: {len(sample_ids)}")
print("Fetching spectrum data...")
print(sample_ids)

print(label_list)
print(data_list)

target_boundaries = 3500 - 100 + 1  # from 100 to 3500 inclusive
for sample_id in sample_ids:
    print(f"Processing sample ID: {sample_id}", end='\r', flush=True)
    intensities, wave_numbers = utils.get_spectrum_data_integer_wavenumbers(sample_id)
    print(f"Intensities length: {len(intensities)}", end='\r', flush=True)
    print(f"Wave numbers length: {len(wave_numbers)}", end='\r', flush=True)
    wavenumber_lens.add(len(wave_numbers))
    min_wavenumber_sample.add(min(wave_numbers))
    max_wavenumber_sample.add(max(wave_numbers))
    print(f"Intensities sample: {intensities[:10]}", end='\r', flush=True)
    print(f"Wave numbers sample: {wave_numbers[:10]}", end='\r', flush=True)
    label=sample_id
    data_list.append(intensities)
    label_list.append(label)
    # when missing datapoints, we pad with zeros at the end
    # we also make a pad layer that will be used in the model to ignore those values
    #for each sample to pad, we must take its minimum and maximum wavenumber into account, and pad up to the interval between 100 and 3500
    # therefore, if the data starts at wavenumber > 100, we must pad at the beginning
    # and if it ends at wavenumber < 3500, we must pad at the end
    # also, if the data exceeds this range, we must truncate it
    current_length = len(intensities)
    if current_length < target_length:
        # Pad with zeros at the end
        padding_length = target_length - current_length
        padded_data = intensities + [0.0] * padding_length
    elif current_length > target_length:
        # Truncate the data
        padded_data = intensities[:target_length]   
    else:
        padded_data = intensities   
    data_list[-1] = padded_data

 
for i in range(len(data_list)):
    current_length = len(data_list[i])
    current_min_wavenumber = data_list[i][0]
    print(f"Current min wavenumber: {current_min_wavenumber}")
    current_max_wavenumber =    data_list[i][-1]
    print(f"Current max wavenumber: {current_max_wavenumber}")
    pad_before = current_min_wavenumber - boundaries[0]
    print(f"Pad before: {pad_before}")
    pad_after = boundaries[1] - current_max_wavenumber
    print(f"Pad after: {pad_after}")
    # Pad with zeros at the beginning and end
     
    data_list[i] = [0.0]*pad_before  + data_list[i] + [0.0]*pad_after 
    # Truncate if necessary
    data_list[i] = data_list[i][:target_length]





print()  # move to a new line after progress updates
print(f"Wavenumber lengths found: {wavenumber_lens}")
print("max wavenumber length:", max(wavenumber_lens))
print(f"wavenumber min ")
print(f"Min wavenumber samples: {min_wavenumber_sample}")
print(f"Max wavenumber samples: {max_wavenumber_sample}")
#make boundaries as a list
boundaries=[min(min_wavenumber_sample), max(max_wavenumber_sample)]
print(f"Overall wavenumber boundaries: {boundaries}")

#print(label_list)
#print(data_list)
# Pad data to have uniform length


        
    
print(f"Data padded to length: {target_length}")
# now 

# Convert lists to tensors
data_tensor = torch.tensor(data_list, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
label_tensor = torch.tensor(label_list, dtype=torch.long)
# Create a TensorDataset and DataLoader
dataset = TensorDataset(data_tensor, label_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# Initialize the model, loss function, and optimizer
num_classes = len(set(label_list))  # Assuming labels are from 0 to num_classes
model = SpectrumClassifier(num_classes)


criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
# Save the trained model


model_cache_dir = 'app/model_cache'
os.makedirs(model_cache_dir, exist_ok=True)
model_path = os.path.join(model_cache_dir, 'spectrum_classifier.pth') 
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')
# this is for fetching all the samples ids and then their data from the database, and finally training the neural network model and storing its cache in app/model_cache/model.pth
# the network is the spectrum classifier defined in spectrum_model.py

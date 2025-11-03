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
from models.spectrum_model import SpectrumClassifier
import os
#import utils
import utils as utils
# Define the path to your database
db_file_path = 'app/database/microplastics_reference.db'

sample_ids=utils.get_all_ids("database/microplastics_reference.db")

data_list=[]
label_list=[]
for sample_id in sample_ids:
    intensities, wave_numbers = utils.get_spectrum_data_integer_wavenumbers(sample_id,"database/microplastics_reference.db")
    label=sample_id
    data_list.append(intensities)
    label_list.append(label)    
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
model_path = os.path.join(model_cache_dir, 'model.pth') 
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')
# this is for fetching all the samples ids and then their data from the database, and finally training the neural network model and storing its cache in app/model_cache/model.pth
# the network is the spectrum classifier defined in spectrum_model.py

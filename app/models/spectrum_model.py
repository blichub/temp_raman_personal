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

import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Define the layers
        self.layer1 = nn.Linear(5, 512)  # Input dimension is 4
        self.layer2 = nn.Linear(512, 128)
        self.layer3 = nn.Linear(128, 32)
        self.layer4 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        
        # Pass through the network layers
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.output(x)
        x = torch.sigmoid(x)  # Apply sigmoid activation to output
        return x
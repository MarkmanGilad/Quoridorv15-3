import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from constants import *

# Parameters
input_size = 246 # state: board = 9*9*3 = 64 + action (2) 
layer1 = 64
output_size = 1 # Q(state, action)
gamma = 0.99 


class DQN (nn.Module):
    def __init__(self, input_channels = 3, row = ROWS, col = COLS, device = None) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) 
        self.MSELoss = nn.MSELoss()

        # Dynamically calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, row, col)
            dummy = self.conv1(dummy)
            dummy = self.conv2(dummy)
            dummy = self.conv3(dummy)
            self.flattened_size = dummy.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.output = nn.Linear(128, 1)      # value
        self.to(self.device)
    
    def forward(self, x: torch.Tensor):
        x = x.to(device=self.device)
        x = F.relu(self.conv1(x))         
        x = F.relu(self.conv2(x))         
        x = F.relu(self.conv3(x))       
        x = x.view(x.size(0), -1)         
        x = F.relu(self.fc1(x)) 
        x = self.output(x)
        return x          
    
    def loss (self, Q_value, rewards, Q_next_Values, dones ):
        rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32).to(device=self.device).unsqueeze(1)
        dones_tensor = torch.tensor(np.array(dones), dtype=torch.int64).to(device=self.device).unsqueeze(1)
        Q_new =  rewards_tensor + gamma * Q_next_Values * (1- dones_tensor)
        return self.MSELoss(Q_value, Q_new)
    
    def load_params(self, path):
        self.load_state_dict(torch.load(path))

    def save_params(self, path):
        torch.save(self.state_dict(), path)

    def copy (self):
        return copy.deepcopy(self)

    def __call__(self, states):
        return self.forward(states)
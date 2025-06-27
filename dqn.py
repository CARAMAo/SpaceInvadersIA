
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.num_units = 128
        self.fc1 = nn.Linear(n_observations, self.num_units)
        # self.layer2 = nn.Linear(self.num_units,  self.num_units)
        # self.layer3 = nn.Linear(self.num_units, self.num_units)
        self.fc2 = nn.Linear(self.num_units,self.num_units)
        self.fc3 = nn.Linear(self.num_units, n_actions)
        
        # self.layer2 = nn.Linear(self.num_units, int(self.num_units/2))
        # self.layer3 = nn.Linear(int(self.num_units/2), int(self.num_units/2))
        # self.layer4 = nn.Linear(int(self.num_units/2), n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.layer3(x))
        return self.fc3(x)
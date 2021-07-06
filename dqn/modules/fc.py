import torch
import torch.nn as nn

class FC1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FC1, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class FC2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(FC2, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

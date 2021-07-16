import torch
import torch.nn as nn

class FC1(nn.Module):
    def __init__(self, in_dim, out_dim, last_layer_activation = True):
        super(FC1, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.last_layer_activation = last_layer_activation

    def forward(self, x):
        x = self.fc(x)
        if self.last_layer_activation:
            x = self.lrelu(x)
        return x

class FC2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, last_layer_activation = True):
        super(FC2, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.last_layer_activation = last_layer_activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.fc2(x)
        if self.last_layer_activation:
            x = self.lrelu(x)
        return x

class FC3(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, last_layer_activation = True):
        super(FC3, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, out_dim, bias=True)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.last_layer_activation = last_layer_activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.fc2(x)
        x = self.lrelu(x)
        x = self.fc3(x)
        if self.last_layer_activation:
            x = self.lrelu(x)
        return x
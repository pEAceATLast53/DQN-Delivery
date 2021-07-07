import torch
import torch.nn as nn

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        BatchNorm = nn.BatchNorm2d
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = BatchNorm(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        #if self.relu is not None:
        #    x = self.relu(x)
        x = self.lrelu(x)
        return x

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.map_size = (args.lidar_range * 2 + 1, args.lidar_range * 2 + 1, 2)
        self.conv1 = BasicConv(2, 8, kernel_size = 3, padding=1)
        self.fc1 = nn.Linear(8 * (self.map_size[0]//2) * (self.map_size[1]//2), 128, bias=True)
        self.fc2 = nn.Linear(128, 128, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.fc2(x)
        x = self.lrelu(x)
        return x

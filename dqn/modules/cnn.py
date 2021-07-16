import torch
import torch.nn as nn
from dqn.modules.fc import FC2

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, bn=True, bias=False):
        super(BasicConv, self).__init__()
        BatchNorm = nn.BatchNorm2d
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = BatchNorm(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.lrelu(x)
        return x

class CNN_FC(nn.Module):
    def __init__(self, height, width, fc_hidden_dim, fc_out_dim):
        super(CNN_FC, self).__init__()
        self.conv1 = BasicConv(2, 8, kernel_size = 3, padding=1)
        self.conv2 = BasicConv(8, 8, kernel_size = 3, padding=1)
        self.fc = FC2(8 * (height//2) * (width//2), fc_hidden_dim, fc_out_dim)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

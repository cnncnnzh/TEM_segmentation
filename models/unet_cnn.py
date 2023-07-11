import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetCNN(nn.Module):
    def __init__(self, n_in, n_out):
        super(UNetCNN, self).__init__()
        self.n_channels = n_in
        self.n_classes = n_out
        self.n_in = n_in
        self.n_out = n_out
        self.relu = nn.ReLU()
        self.in_cov = nn.Conv2d(n_in, 64, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_6 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(64, n_out, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (1,1,256,256)
        x1 = self.in_cov(x)  # x1: (1,64,256,256)

        # first layer
        x2 = self.relu(self.conv_1(x1))  # x2: (1,128,256,256)

        # second layer
        x3 = self.relu(self.conv_2(x2))  # x3: (1, 256, 256, 256)

        # 3rd laryer
        x4 = self.relu(self.conv_3(x3))  # x4:(1, 512, 256, 256)

        # 4th layer
        x5 = self.relu(self.conv_4(x4))  # x5:(1, 256, 256,256)

        # 5th layer
        x6 = self.relu(self.conv_5(x5))  # x6:(1,128,256,256)

        # 6th layer
        x7 = self.relu(self.conv_6(x6))  # x7:(1,64,256,256)

        # output
        x_out = self.out_conv(x7)  # x_out: (1,n_out, 256,256)

        return x_out

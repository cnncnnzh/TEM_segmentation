import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetWOSkip(nn.Module):
    def __init__(self, n_in, n_out):
        super(UNetWOSkip, self).__init__()
        self.n_channels = n_in
        self.n_classes = n_out
        self.n_in = n_in
        self.n_out = n_out
        self.relu = nn.ReLU()
        self.in_cov = nn.Conv2d(n_in, 64, kernel_size=3, padding=1)
        self.maxpool_conv = nn.MaxPool2d(2)
        self.conv_down1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_down2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv_down3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_up2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_up3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(64, n_out, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (1,1,256,256)
        x1 = self.in_cov(x)  # x1: (1,64,256,256)
        # downsampling layer 1_maxpooling
        x2 = self.maxpool_conv(x1)  # x2: (1, 64, 128, 128)

        # downsampling layer 1_conv, ReLU as activation functon
        x3 = self.relu(self.conv_down1(x2))  # x3: (1, 128, 128, 128)

        # downsampling layer 2_maxpooling
        x4 = self.maxpool_conv(x3)  # x4: (1, 128, 64, 64)
        # downsampling layer 2_conv
        x5 = self.relu(self.conv_down2(x4))  # x5: (1, 256, 64, 64)

        # downsampling layer 3_maxpooling
        x6 = self.maxpool_conv(x5)  # x6:(1,256,32,32)

        # downsampling layer 3_conv
        x7 = self.relu(self.conv_down3(x6))  # x7(1,512,32,32)

        # upsampling layer 1_up
        x8 = self.up(x7)  # x7:(1,512,64,64)

        # Upsampling1_conv
        x9 = self.relu(self.conv_up1(x8))  # x9: (1,256,64,64)

        # Upsampling layer 2_up
        x10 = self.up(x9)  # x10: (1,256,128,128)

        # Upsampling2_conv
        x11 = self.relu(self.conv_up2(x10))  # x11: (1,128,128,128)

        # Upsampling layer 3_up
        x12 = self.up(x11)  # x12: (1,128,256,256)

        # Upsampling3_conv
        x13 = self.relu(self.conv_up3(x12))  # x13: (1,64,256,256)

        # output
        x_out = self.out_conv(x13)  # x_out: (1,n_out, 256,256)

        return x_out

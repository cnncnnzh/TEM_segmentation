import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet2Layer(nn.Module):
    def __init__(self, n_in, n_out):
        super(UNet2Layer, self).__init__()
        self.n_channels = n_in
        self.n_classes = n_out
        self.n_in = n_in
        self.n_out = n_out
        self.relu = nn.ReLU()
        self.in_cov = nn.Conv2d(n_in, 64, kernel_size=3, padding=1)
        self.maxpool_conv = nn.MaxPool2d(2)
        self.conv_down1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_down2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = nn.Conv2d(384, 128, kernel_size=3, padding=1)
        self.conv_up2 = nn.Conv2d(192, 64, kernel_size=3, padding=1)
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
        # Upsampling layer 1_up
        x6 = self.up(x5)  # x6: (1, 256, 128, 128)

        # skip connection_1
        w_diff = x3.size()[2] - x6.size()[2]  # w_diff=0
        h_diff = x3.size()[3] - x6.size()[3]  # h_diff=0

        x6 = F.pad(x6, [h_diff // 2, h_diff - h_diff // 2,
                        w_diff // 2, w_diff - w_diff // 2])  # x6: (1, 256, 128, 128)
        x = torch.cat([x3, x6], dim=1)  # x: (1, 384, 128, 128)

        # Upsampling1_conv
        x7 = self.relu(self.conv_up1(x))  # x7: (1, 128, 128, 128)

        # Upsampling layer 1_up
        x8 = self.up(x7)  # x8: (1, 128, 256, 256)
        # skip connection_2
        w_diff_2 = x1.size()[2] - x8.size()[2]  # w_diff_2=0
        h_diff_2 = x1.size()[3] - x8.size()[3]  # h_diff_2=0

        x8 = F.pad(x8, [h_diff_2 // 2, h_diff_2 - h_diff_2 // 2,
                        w_diff_2 // 2, w_diff_2 - w_diff_2 // 2])
        x = torch.cat([x1, x8], dim=1)  # x: (1, 192, 256, 256)

        ##Upsampling2_conv, ReLU as activation functon
        x9 = self.relu(self.conv_up2(x))  # x9: (1, 64, 256, 256)

        x_out = self.out_conv(x9)  # x_out: (1, 1, 256, 256)
        return x_out

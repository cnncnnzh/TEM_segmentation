import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3Layer(nn.Module):
    def __init__(self, n_in, n_out):
        super(UNet3Layer, self).__init__()
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
        self.conv_up1 = nn.Conv2d(768, 256, kernel_size=3, padding=1)
        self.conv_up2 = nn.Conv2d(384, 128, kernel_size=3, padding=1)
        self.conv_up3 = nn.Conv2d(192, 64, kernel_size=3, padding=1)
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

        # skip conection_1
        w_diff = x5.size()[2] - x8.size()[2]  # w_diff=0
        h_diff = x5.size()[3] - x8.size()[3]  # h_diff=0

        x8 = F.pad(x8, [h_diff // 2, h_diff - h_diff // 2,
                        w_diff // 2, w_diff - w_diff // 2])  # x8: (1, 512, 64, 64)
        x = torch.cat([x5, x8], dim=1)  # x: (1, 768,64,64)

        # Upsampling1_conv
        x9 = self.relu(self.conv_up1(x))  # x9: (1,256,64,64)

        # Upsampling layer 2_up
        x10 = self.up(x9)  # x10: (1,256,128,128)

        # Skip connection
        w_diff = x3.size()[2] - x10.size()[2]  # w_diff=0
        h_diff = x3.size()[3] - x10.size()[3]  # h_diff=0

        x10 = F.pad(x10, [h_diff // 2, h_diff - h_diff // 2,
                          w_diff // 2, w_diff - w_diff // 2])  # x10: (1, 256, 128, 128)
        x = torch.cat([x3, x10], dim=1)  # x: (1, 384,128,128)

        # Upsampling2_conv
        x11 = self.relu(self.conv_up2(x))  # x11: (1,128,128,128)

        # Upsampling layer 3_up
        x12 = self.up(x11)  # x12: (1,128,256,256)

        # Skip connection
        w_diff = x1.size()[2] - x12.size()[2]  # w_diff=0
        h_diff = x1.size()[3] - x12.size()[3]  # h_diff=0

        x12 = F.pad(x12, [h_diff // 2, h_diff - h_diff // 2,
                          w_diff // 2, w_diff - w_diff // 2])  # x12: (1, 128, 256, 256)
        x = torch.cat([x1, x12], dim=1)  # x: (1, 192,256,256)

        # Upsampling3_conv
        x13 = self.relu(self.conv_up3(x))  # x13: (1,64,256,256)

        # output
        x_out = self.out_conv(x13)  # x_out: (1,n_out, 256,256)

        return x_out

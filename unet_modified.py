import torch
import torch.nn as nn
from Unet_Module.down_convolution import DownConvolution
from Unet_Module.up_convolution import UpConvolution
from Unet_Module.last_conv_block import LastConvolution
from Unet_Module.simple_double_conv import SimpleConvolution



class UNetModern(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(UNetModern, self).__init__()
        self.simpleConv = SimpleConvolution(input_channel, 64)
        self.downConvBock1 = DownConvolution(64, 128)
        self.downConvBock2 = DownConvolution(128, 256)
        self.downConvBock3 = DownConvolution(256, 512)
        self.midMaxpool = nn.MaxPool2d(2, 2)
        self.upConvBlock0 = UpConvolution(512, 1024)
        self.upConvBlock1 = UpConvolution(1024, 512)
        self.upConvBlock2 = UpConvolution(512, 256)
        self.upConvBlock3 = UpConvolution(256, 128)
        self.lastConv = LastConvolution(128, 64, num_classes)

    def forward(self, x):
        x_1 = self.simpleConv(x)  #skip_x_1
        x_2 = self.downConvBock1(x_1)  # skip_x_2
        x_3 = self.downConvBock2(x_2)  # skip_x_3
        x_4 = self.downConvBock3(x_3)  # skip_x_4
        x_5 = self.midMaxpool(x_4)
        x_6 = self.upConvBlock0(x_5)
        concat_x_4_6 = torch.cat((x_4, x_6), 1)
        x_7 = self.upConvBlock1(concat_x_4_6)
        concat_x_3_7 = torch.cat((x_3, x_7), 1)
        x_8 = self.upConvBlock2(concat_x_3_7)
        concat_x_2_8 = torch.cat((x_2, x_8), 1)
        x_9 = self.upConvBlock3(concat_x_2_8)
        concat_x_1_9 = torch.cat((x_1, x_9), 1)
        out = self.lastConv(concat_x_1_9)

        return out


if __name__ == "__main__":
    unet_modern = UNetModern(1, 2)
    inp = torch.rand(4, 1, 128, 128)
    out = unet_modern(inp)
    # Must output (batch_size, num_classes, w, h)
    # (4,2,388,388)
    print(out.size())

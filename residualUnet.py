import torch
import torch.nn as nn
from Unet_Module.residual_down_conv_block import DownResBlock
from Unet_Module.residual_up_conv_block import UpResConvolution
from Unet_Module.residual_conv_block import SimpleResBlock



class ResidualUnet(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(ResidualUnet, self).__init__()
        self.simpleResConv = SimpleResBlock(input_channel, 64)
        self.downResConvBock1 = DownResBlock(64, 128,(2,2))
        self.downResConvBock2 = DownResBlock(128, 256,(2,2))
        self.downResConvBock3 = DownResBlock(256, 512,(2,2))
        self.upResConvBlock1 = UpResConvolution(512, 256)
        self.upResConvBlock2 = UpResConvolution(256, 128)
        self.upResConvBlock3 = UpResConvolution(128, 64)
        self.lastConv = nn.Conv2d(64, num_classes,(1,1))

    def forward(self, x):
        x_1 = self.simpleResConv(x)  #skip_x_1
        x_2 = self.downResConvBock1(x_1)  # skip_x_2
        x_3 = self.downResConvBock2(x_2)  # skip_x_3
        x_4 = self.downResConvBock3(x_3)  # skip_x_4
        x_5 = self.upResConvBlock1(x_4,x_3)
        x_6 = self.upResConvBlock2(x_5,x_2)
        x_7 = self.upResConvBlock3(x_6,x_1)
        out = self.lastConv(x_7)

        return out


if __name__ == "__main__":
    res_unet = ResidualUnet(1, 2)
    inp = torch.rand(4, 1, 128, 128)
    out = res_unet(inp)
    # Must output (batch_size, num_classes, w, h)
    # (4,2,128,128)
    print(out.size())

import torch
import torch.nn as nn
from Unet_Module.down_convolution import DownConvolution
from Unet_Module.up_convolution import UpConvolution
from Unet_Module.last_conv_block import LastConvolution
from Unet_Module.simple_double_conv import SimpleConvolution
from Unet_Module.attention_module import UnetAttentionModule


class AttentionUNet(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(AttentionUNet, self).__init__()
        self.simpleConv = SimpleConvolution(input_channel, 64)
        self.downConvBock1 = DownConvolution(64, 128)
        self.downConvBock2 = DownConvolution(128, 256)
        self.downConvBock3 = DownConvolution(256, 512)
        self.attnBlock1 = UnetAttentionModule(512,256)
        self.attnBlock2 = UnetAttentionModule(256, 128)
        self.attnBlock3 = UnetAttentionModule(128, 64)
        self.upConvBlock0 = UpConvolution(512, 512)
        self.upConvBlock1 = UpConvolution(512, 256)
        self.upConvBlock2 = UpConvolution(256, 128)
        self.lastConv = LastConvolution(128, 64, num_classes)

    def forward(self, x):
        x_1 = self.simpleConv(x)  # attn_x_1
        x_2 = self.downConvBock1(x_1)  # attn_x_2
        x_3 = self.downConvBock2(x_2)  # attn_x_3
        x_4 = self.downConvBock3(x_3)  # attn_x_4
        x_6 = self.upConvBlock0(x_4)
        x_3_4_attention = self.attnBlock1(x_4,x_3)
        concat_x_4_6 = torch.cat((x_3_4_attention, x_6), 1)
        x_7 = self.upConvBlock1(concat_x_4_6)
        x_3_attention = self.attnBlock2(x_3, x_7)
        concat_x_3_7 = torch.cat((x_3_attention, x_7), 1)
        x_8 = self.upConvBlock2(concat_x_3_7)
        x_2_attention = self.attnBlock3(x_2, x_8)
        concat_x_2_8 = torch.cat((x_2_attention, x_8), 1)
        out = self.lastConv(concat_x_2_8)

        return out


if __name__ == "__main__":
    attnunet = AttentionUNet(1, 2)
    inp = torch.rand(4, 1, 128, 128)
    out = attnunet(inp)
    # Must output (batch_size, num_classes, w, h)
    # (4,2,388,388)
    print(out.size())

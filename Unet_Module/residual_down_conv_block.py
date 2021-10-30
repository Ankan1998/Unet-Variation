import torch
import torch.nn as nn
from helper.conv_same_padding import Conv2dSamePadding


class DownResBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(DownResBlock, self).__init__()
        self.conv1 = Conv2dSamePadding(input_channel, output_channel, (3, 3),(2,2))
        self.conv2 = Conv2dSamePadding(output_channel, output_channel, (3, 3))
        self.shortcut = Conv2dSamePadding(input_channel, output_channel, (1, 1),(2,2))
        self.batchnorm1 = nn.BatchNorm2d(input_channel)
        self.batchnorm2 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_Identity = self.shortcut(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = x + x_Identity
        return out


if __name__ == "__main__":
    down_res = DownResBlock(64, 128)
    inp = torch.rand(4, 64, 256, 256)
    out = down_res(inp)
    print(out.size())

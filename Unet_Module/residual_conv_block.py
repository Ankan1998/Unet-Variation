import torch
import torch.nn as nn
from helper.conv_same_padding import Conv2dSamePadding


class SimpleResBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(SimpleResBlock, self).__init__()
        self.conv1 = Conv2dSamePadding(input_channel, output_channel, (3, 3))
        self.conv2 = Conv2dSamePadding(output_channel, output_channel, (3, 3))
        self.shortcut = Conv2dSamePadding(input_channel, output_channel, (1, 1))
        self.batchnorm = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_Identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = x + x_Identity
        return out


if __name__ == "__main__":
    simple_res = SimpleResBlock(1, 64)
    inp = torch.rand(4, 1, 128, 128)
    out = simple_res(inp)
    print(out.size())

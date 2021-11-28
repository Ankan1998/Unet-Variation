import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetAttentionModule(nn.Module):
    def __init__(self, g_input_channel, x_input_channel):
        super(UnetAttentionModule, self).__init__()
        self.phi_g = nn.Conv2d(g_input_channel, 128, (1, 1), (1, 1))
        self.theta_x = nn.Conv2d(x_input_channel, 128, (1, 1), (2, 2))
        self.relu = nn.ReLU()
        self.one_filter = nn.LazyConv2d(1, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g_conv = self.phi_g(g)
        x_conv = self.theta_x(x)
        aligned_weight_summed = g_conv + x_conv
        aligned_weight_summed_relu = self.relu(aligned_weight_summed)
        aligned_weight_summed_relu_one_filter = self.one_filter(aligned_weight_summed_relu)
        attention_coeff = self.sigmoid(aligned_weight_summed_relu_one_filter)
        attention_upsampled = F.interpolate(attention_coeff, (x.size(2), x.size(3)))
        # expand the attention upsampled to x channel dimension
        attention_upsampled_expanded = attention_upsampled.expand(x.size(0), x.size(1), x.size(2), x.size(3))
        out = x * attention_upsampled_expanded

        return out


if __name__ == "__main__":
    unetAttention = UnetAttentionModule(64, 32)
    g = torch.rand(4, 64, 32, 32)
    x = torch.rand(4, 32, 64, 64)
    # output --> 4,32,64,64
    res = unetAttention(g, x)
    print(res.size())

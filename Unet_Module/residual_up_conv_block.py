import torch
import torch.nn as nn
from helper.conv_same_padding import Conv2dSamePadding
from Unet_Module.residual_down_conv_block import DownResBlock
# 1. 2 Convolution[(3x3),Relu] --> TransposeConv(out_channel,out_channel//2)[kernel = 2x2 , stride =2x2]


class UpResConvolution(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(UpResConvolution, self).__init__()
        self.convtranspose = nn.ConvTranspose2d(input_channel,input_channel//2,(2,2),(2,2))
        self.resblock = DownResBlock(input_channel,output_channel)


    def forward(self, x,x_skip):
        x = self.convtranspose(x)
        x_Identity = torch.cat((x,x_skip),1)
        out = self.resblock(x_Identity)

        return out

if __name__=="__main__":
    up_res = UpResConvolution(1024,512)
    inp = torch.rand(4,1024,32,32)
    x_skip = torch.rand(4,512,64,64)
    out = up_res(inp,x_skip)
    print(out.size())



##From: https://github.com/janesjanes/Pytorch-TextureGAN/blob/master/models/texturegan.py
import torch
import torch.nn as nn


class TextureGAN(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        """
        Defines the necessary modules of the TextureGAN Generator

        Input:
        - int input_nc : Input number of channels
        - int output_nc : Output number of channels
        """
        super(TextureGAN, self).__init__()

        self.conv = nn.Conv2d
        self.batch_norm = nn.BatchNorm2d
        self.ngf = ngf
        self.input_nc = input_nc
        self.output_nc = output_nc

        self.res_block = ResidualBlock
        self.biup = UpsamplingBlock
        self.main_model = MainModel
        self.model = self.create_model()

    def create_model(self):
        skip_block = nn.Sequential()

        skip_block.add_module('main_model', self.main_model(self.input_nc, self.output_nc, self.ngf))
        skip_block.add_module('conv_6', self.conv(self.ngf+5, self.ngf*2, 3, 1, 1))
        skip_block.add_module('res_block_14', self.res_block(self.ngf*2,self.ngf*2))
        skip_block.add_module('res_block_15', self.res_block(self.ngf*2,self.ngf*2))
        skip_block.add_module('conv_7', self.conv(self.ngf*2, 3, 3, 1, 1))
        skip_block.add_module('batch_9', self.batch_norm(3))

        return skip_block
    
    def forward(self, x):
        return self.model(x)


class MainModel(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        """
        Function which pieces together the model
        """
        super(MainModel, self).__init__()
        self.conv = nn.Conv2d
        self.batch_norm = nn.BatchNorm2d
        self.ngf = ngf
        self.input_nc = input_nc
        self.output_nc = output_nc

        self.res_block = ResidualBlock
        self.biup = UpsamplingBlock
        model = nn.Sequential()
        
        model.add_module('conv_1', self.conv(input_nc,ngf,3,1,1))
        model.add_module('batch_1', self.batch_norm(ngf))
        model.add_module('norm_1', nn.ReLU(True))

        model.add_module('res_block_1', self.res_block(ngf,ngf))
        model.add_module('conv_2', self.conv(ngf,ngf*2,3,2,1))
        model.add_module('batch_2',self.batch_norm(ngf*2))
        model.add_module('norm_2', nn.ReLU(True))

        model.add_module('res_block_2', self.res_block(ngf*2,ngf*2))

        model.add_module('conv_3',self.conv(ngf*2,ngf*4,3,2,1))
        model.add_module('batch_3',self.batch_norm(ngf*4))
        model.add_module('norm_3',nn.ReLU(True))

        model.add_module('res_block_3',self.res_block(ngf*4,ngf*4))

        model.add_module('conv_4',self.conv(ngf*4,ngf*8,3,2,1))
        model.add_module('batch_4',self.batch_norm(ngf*8))
        model.add_module('norm_4',nn.ReLU(True))
        
        model.add_module('res_block_4',self.res_block(ngf*8,ngf*8))
        model.add_module('res_block_5',self.res_block(ngf*8,ngf*8))
        model.add_module('res_block_6',self.res_block(ngf*8,ngf*8))
        model.add_module('res_block_7',self.res_block(ngf*8,ngf*8))
        model.add_module('res_block_8',self.res_block(ngf*8,ngf*8))

        model.add_module('upsampl_1',self.biup(ngf*8,ngf*4,3,1,1))
        model.add_module('batch_5',self.batch_norm(ngf*4))
        model.add_module('norm_5',nn.ReLU(True))
        model.add_module('res_block_9',self.res_block(ngf*4,ngf*4))
        model.add_module('res_block_10',self.res_block(ngf*4,ngf*4))

        model.add_module('upsampl_2',self.biup(ngf*4,ngf*2,3,1,1))
        model.add_module('batch_6',self.batch_norm(ngf*2))
        model.add_module('norm_6',nn.ReLU(True))
        model.add_module('res_block_11',self.res_block(ngf*2,ngf*2))
        model.add_module('res_block_12',self.res_block(ngf*2,ngf*2))

        model.add_module('upsampl_3',self.biup(ngf*2,ngf,3,1,1))
        model.add_module('batch_7',self.batch_norm(ngf))
        model.add_module('norm_7',nn.ReLU(True))
        model.add_module('res_block_13',self.res_block(ngf,ngf))
        model.add_module('batch_8',self.batch_norm(ngf))

        self.main_model = model


    def forward(self, x):
        return torch.cat((self.main_model(x), x), 1)
        #return self.main_model(input)


class UpsamplingBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel, stride, pad):
        """
        Single block of upsampling operation

        Input:
        - int input_nc    : Input number of channels
        - int output_nc   : Output number of channels
        - int kernel      : Kernel size
        - int stride	  : Stride length
        - int pad         : Padd_moduleing
        """
        super(UpsamplingBlock, self).__init__()

        conv = nn.Conv2d
        biup = nn.Upsample

        block = nn.Sequential()
        block.add_module('conv_1', conv(input_nc, output_nc, kernel, stride, pad))
        block.add_module('upsample_2', biup(scale_factor=2, mode='bilinear'))

        self.biup_block = block

    def forward(self, x):
        return self.biup_block(x)


# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=padding, dilation=dilation)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = conv3x3(in_channels, out_channels, stride,
                             padding=dilation[0])
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride, 
                             padding=dilation[1])
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out

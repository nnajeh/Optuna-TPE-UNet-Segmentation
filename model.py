# Import libraries
from libraries import *


import torch
import torch.nn as nn

# Function to adjust filter sizes based on `n_filters`
def double_convolution(in_channels, out_channels):
    """
    Two consecutive convolutional layers followed by BatchNorm and ReLU activation.
    """
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return conv_op

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_filters=64):
        super(UNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Contracting path (scaled based on `n_filters`)
        self.down_convolution_1 = double_convolution(in_channels, n_filters)
        self.down_convolution_2 = double_convolution(n_filters, n_filters * 2)
        self.down_convolution_3 = double_convolution(n_filters * 2, n_filters * 4)
        self.down_convolution_4 = double_convolution(n_filters * 4, n_filters * 8)
        self.down_convolution_5 = double_convolution(n_filters * 8, n_filters * 16)

        # Expanding path (scaled based on `n_filters`)
        self.up_transpose_1 = nn.ConvTranspose2d(in_channels=n_filters * 16, out_channels=n_filters * 8, kernel_size=2, stride=2)
        self.up_convolution_1 = double_convolution(n_filters * 16, n_filters * 8)

        self.up_transpose_2 = nn.ConvTranspose2d(in_channels=n_filters * 8, out_channels=n_filters * 4, kernel_size=2, stride=2)
        self.up_convolution_2 = double_convolution(n_filters * 8, n_filters * 4)

        self.up_transpose_3 = nn.ConvTranspose2d(in_channels=n_filters * 4, out_channels=n_filters * 2, kernel_size=2, stride=2)
        self.up_convolution_3 = double_convolution(n_filters * 4, n_filters * 2)

        self.up_transpose_4 = nn.ConvTranspose2d(in_channels=n_filters * 2, out_channels=n_filters, kernel_size=2, stride=2)
        self.up_convolution_4 = double_convolution(n_filters * 2, n_filters)

        # Output layer
        self.out = nn.Conv2d(in_channels=n_filters, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # Contracting path
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)
        
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)
        
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)
        
        down_9 = self.down_convolution_5(down_8)

        # Expanding path
        up_1 = self.up_transpose_1(down_9)
        up_2 = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        
        up_3 = self.up_transpose_2(up_2)
        up_4 = self.up_convolution_2(torch.cat([down_5, up_3], 1))
        
        up_5 = self.up_transpose_3(up_4)
        up_6 = self.up_convolution_3(torch.cat([down_3, up_5], 1))
        
        up_7 = self.up_transpose_4(up_6)
        up_8 = self.up_convolution_4(torch.cat([down_1, up_7], 1))

        # Final output layer
        out = self.out(up_8)

        return out

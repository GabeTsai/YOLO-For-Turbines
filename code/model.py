import torch
import torch.nn as nn

"""
C. @aladdinpersson

Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer

"""

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1], #each residual block downsamples the feature map then upsamples it by a factor of 2
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
    """
    One CNN block for YOLOv3. Conv2d -> BatchNorm2d -> LeakyReLU or just Conv2d.

    Args:
        in_channels: int, number of input channels
        out_channels: int, number of output channels/filters
        batch_norm_act: bool, whether to use batch normalization and activation function
        **kwargs: additional conv layer settings
    
    """
    def __init__(self, in_channels, out_channels, batch_norm_act = True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias = not batch_norm_act, **kwargs) #using batch norm eliminates need for bias vector
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.batch_norm_act = batch_norm_act

    def forward(self, x):
        if self.batch_norm_act:
            return self.leaky_relu(self.batch_norm(self.conv(x)))
        else:
            return self.conv(x) 
        
class ResidualBlock(nn.Module):
    """
    One residual block for YOLOv3. Consists of two CNN blocks with optional residual connection.
    One block reduces feature map dimensionality by 2, forcing network to retain most important features.
    The other expands it back to the original size, incorporating additional context and details with refined features from compression phase.

    Args:
        in_channels: int, number of input channels
        use_residual: bool, whether to use residual connection
        num_blocks: int, number of residual blocks to stack
    """
    def __init__(self, in_channels, use_residual = True, num_blocks = 1):
        super().__init__()
        self.layers = nn.ModuleList()
        for block in range(num_blocks):
            self.layers += [
                nn.Sequential(
                    CNNBlock(in_channels, in_channels//2, kernel_size = 1),     #reduce dimensionality
                    CNNBlock(in_channels//2, in_channels, kernel_size = 3, padding = 1)     #expand feature map for more feature refinement
                )   
            ]
        self.use_residual = use_residual
        self.num_blocks = num_blocks
    
    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x) 
            else:
                x = layer(x)
        return x

class ScalePredictionBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        

class Yolov3(nn.Module):
    pass
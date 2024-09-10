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
    ["B", 8],   #route to detection head
    (512, 3, 2),
    ["B", 8],   #route to detection head
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
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm_act else None
        self.leaky_relu = nn.LeakyReLU(0.1) if batch_norm_act else None
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
                    CNNBlock(in_channels//2, in_channels, kernel_size = 3, padding = 1)     #expand feature map 
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
    """
    Detection head for YOLOv3. 
    Output is a tensor with shape (B, 3, scale_dim, scale_dim, tx + ty + tw + th + objectness + class_scores)
    
    Args:
        in_channels: int, number of input channels
        num_classes: int, number of classes to predict
        anchors_per_scale: int, number of anchors per scale
    """
    def __init__(self, in_channels, num_classes, anchors_per_scale = 3):
        super().__init__()
        self.pred_block = nn.Sequential(
            CNNBlock(in_channels, in_channels * 2, kernel_size = 3, padding = 1),
            CNNBlock(2 * in_channels, (num_classes + 5) * anchors_per_scale, batch_norm_act = False, kernel_size = 1)
        )
        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale

    def forward(self, x):
        x = self.pred_block(x) #(B, 3 * (num_classes + 5), scale_dim, scale_dim)
        x = x.reshape(x.shape[0], self.anchors_per_scale, self.num_classes + 5, x.shape[2], x.shape[3])
        return x.permute(0, 1, 3, 4, 2) #(B, 3, scale_dim, scale_dim, num_classes + 5)

class YOLOv3(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 80):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = self._create_model_layers()

    def forward(self, x):
        predictions = []
        route_to_detection_head = []

        for layer in self.layers:
            if isinstance(layer, ScalePredictionBlock):
                predictions.append(layer(x))
                continue
                
            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_blocks == 8:
                route_to_detection_head.append(x)
            
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_to_detection_head[-1]], dim = 1) #concatenate by channel dimension
                route_to_detection_head.pop() 

        return predictions
    
    def _create_model_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for block in config:
            if isinstance(block, tuple):
                out_channels, kernel_size, stride = block
                padding = 1 if kernel_size == 3 else 0
                layers.append(
                    CNNBlock(in_channels, out_channels, kernel_size = kernel_size, 
                             stride = stride, padding = padding)
                )
                in_channels = out_channels

            elif isinstance(block, list): #residual blocks do not change feature map dimensions
                layers.append(ResidualBlock(in_channels, num_blocks = block[1]))
            
            #Detection head. 5 conv layers, alternating 1x1 and 3x3 
            elif isinstance(block, str): 
                if block == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual = False, num_blocks = 1),
                        CNNBlock(in_channels, in_channels//2, kernel_size = 1),
                        ScalePredictionBlock(in_channels//2, num_classes = self.num_classes)
                    ]
                    in_channels = in_channels // 2 

                elif block == "U":
                    layers.append(nn.Upsample(scale_factor = 2))
                    in_channels = in_channels * 3   #since we concatenate with a feature map with twice the number of channels
        return layers

if __name__ == "__main__":
    model = YOLOv3()
    print(sum(p.numel() for p in model.parameters()))
    with open("yolov3_summary.txt", "w") as f:
        f.write(str(model))
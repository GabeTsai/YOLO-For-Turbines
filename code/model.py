import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

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
    
    def set_layers(self, layers):
        self.conv = layers[0]
        if self.batch_norm_act:
            self.batch_norm = layers[1]
            self.leaky_relu = layers[2]

    def forward(self, x):
        if self.batch_norm_act:
            # print(torch.sum(torch.isnan(self.conv(x))))
            # print(self.batch_norm(self.conv(x)))
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
    
    def set_layers(self, layers):
        self.layers = layers

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

    def set_layers(self, layers):
        self.pred_block = layers

    def forward(self, x):
        x = self.pred_block(x) #(B, 3 * (num_classes + 5), scale_dim, scale_dim)
        x = x.reshape(x.shape[0], self.anchors_per_scale, self.num_classes + 5, x.shape[2], x.shape[3])
        return x.permute(0, 1, 3, 4, 2) #(B, 3, scale_dim, scale_dim, num_classes + 5)

class YOLOv3(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 80, weights_path = None):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = self._create_model_layers()
        self.param_idx = 0
        self.layer_id = 0
        self.weights_path = None

        if weights_path:
            self.weights_path = weights_path
            with open (weights_path, "rb") as f:
                header = np.fromfile(f, dtype = np.int32, count = 5)
                self.weights = np.fromfile(f, dtype = np.float32)
            self.cutoff = None
            file_name = os.path.basename(weights_path)
            if '.conv' in weights_path:
                self.cutoff = int(file_name.split('.')[-1])

    def forward(self, x):
        predictions = []
        route_to_detection_head = []
        assert torch.sum(torch.isnan(x)) == 0
        for layer in self.layers:
            if isinstance(layer, ScalePredictionBlock):
                predictions.append(layer(x))
                continue

            x = layer(x)
            
            if torch.sum(torch.isnan(x)) > 0:
                raise ValueError("Nan in layer")
            
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
    
    def load_weights(self):        
        for i, block in enumerate(self.layers):
            if isinstance(block, CNNBlock):
                self.layers[i] = self.load_CNNBlock(block)  
            #residual block or scale prediction block
            elif isinstance(block, ResidualBlock) or isinstance(block, ScalePredictionBlock):
                self.layers[i] = self.load_block_weights(block)
            else:
                self.layers[i] = self.load_layer_weights(block)
        print(f"Weights from {self.weights_path} loaded successfully.")
    
    def load_CNNBlock(self, block):
        loaded_block = block
        cnn_block_layers = []
        if block.batch_norm_act:
            loaded_batch_norm = self.load_layer_weights(block.batch_norm)
            loaded_conv = self.load_layer_weights(block.conv)
            cnn_block_layers += [loaded_conv, loaded_batch_norm, block.leaky_relu]
        #just a single conv layer
        else:   
            loaded_conv = self.load_layer_weights(block.conv)
            cnn_block_layers.append(loaded_conv)
        loaded_block.set_layers(cnn_block_layers)
        return loaded_block
    
    def load_block_weights(self, block):
        loaded_block = block

        if isinstance(block, ResidualBlock):
            loaded_res_layers = nn.ModuleList()
            res_layers = block.layers
            for seq in res_layers:
                loaded_seq_layers = []
                for cnn_block in seq.children():
                    loaded_seq_layers.append(self.load_CNNBlock(cnn_block))
                loaded_res_layers.append(nn.Sequential(*loaded_seq_layers))
            loaded_block.set_layers(loaded_res_layers)

        elif isinstance(block, ScalePredictionBlock):
            loaded_scale_blocks = []
            for seq in block.children():
                for cnn_block in seq.children():
                    loaded_scale_blocks.append(self.load_CNNBlock(cnn_block))
            loaded_block.set_layers(nn.Sequential(*loaded_scale_blocks))
        return loaded_block
    
    def load_layer_weights(self, layer):
        if self.layer_id == self.cutoff:
            return layer
        weights = self.weights
        if isinstance(layer, nn.Conv2d):
            if layer.bias is not None: #if bias exists
                num_bias = layer.bias.numel()
                conv_biases = torch.from_numpy(weights[self.param_idx:self.param_idx + num_bias]).float()
                conv_biases = conv_biases.view_as(layer.bias)
                self.param_idx += num_bias
                layer.bias.data.copy_(conv_biases)

            num_weights = layer.weight.numel()
            conv_weights = torch.from_numpy(weights[self.param_idx:self.param_idx + num_weights]).float()
            self.param_idx += num_weights
            conv_weights = conv_weights.view_as(layer.weight)
            layer.weight.data.copy_(conv_weights)

        elif isinstance(layer, nn.BatchNorm2d):
            num_bn_params = layer.bias.numel()
            
            bn_biases = torch.from_numpy(weights[self.param_idx:self.param_idx + num_bn_params]).float()  #beta
            layer.bias.data.copy_(bn_biases.view_as(layer.bias))
            self.param_idx += num_bn_params

            bn_weights = torch.from_numpy(weights[self.param_idx:self.param_idx + num_bn_params]).float() #gamma
            layer.weight.data.copy_(bn_weights.view_as(layer.weight))        
            self.param_idx += num_bn_params

            bn_running_mean = torch.from_numpy(weights[self.param_idx:self.param_idx + num_bn_params]).float()
            layer.running_mean.data.copy_(bn_running_mean.view_as(layer.running_mean))
            self.param_idx += num_bn_params

            bn_running_var = torch.from_numpy(weights[self.param_idx:self.param_idx + num_bn_params]).float()
            layer.running_var.data.copy_(bn_running_var.view_as(layer.running_var))
            self.param_idx += num_bn_params

        self.layer_id += 1
        return layer

if __name__ == "__main__":
    num_classes = 80
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    
    print(sum(p.numel() for p in model.parameters()))

    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")
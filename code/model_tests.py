import torch
from model import CNNBlock, ResidualBlock, ScalePredictionBlock

IMAGE_SIZE = 416

def test_CNNBlock():
    """
    Make sure output shape of CNNBlock is as expected.
    """
    in_channels = 3
    out_channels = 32
    block = CNNBlock(in_channels, out_channels, kernel_size = 1)
    x = torch.randn((5, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = block(x)
    assert out.shape == (5, 32, IMAGE_SIZE, IMAGE_SIZE)

def test_ResidualBlock():
    """
    Make sure output shape of ResidualBlock is as expected.
    Test example below is for the second residual block in the network.
    """
    in_channels = 128
    block = ResidualBlock(in_channels, num_blocks = 2)
    feature_map_dim = 64
    x = torch.randn((5, in_channels, feature_map_dim, feature_map_dim))
    out = block(x)
    assert out.shape == (5, in_channels, feature_map_dim, feature_map_dim)

def test_ScalePredictionBlock():
    in_channels = 512
    feature_map_dim = 13
    block = ScalePredictionBlock(in_channels, num_classes = 2)
    x = torch.randn((5, in_channels, feature_map_dim, feature_map_dim))
    out = block(x)
    assert out.shape == (5, 3, feature_map_dim, feature_map_dim, 5 + 2)

def main():
    test_CNNBlock()
    test_ResidualBlock()
    test_ScalePredictionBlock()
    print("All tests passed.")

if __name__ == "__main__":
    main()
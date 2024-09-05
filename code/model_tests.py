import torch
from model import CNNBlock, ResidualBlock, ScalePredictionBlock, YOLOv3
from loss import YOLOLoss

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

def test_Yolov3():
    num_classes = 420
    model = YOLOv3(num_classes = num_classes)
    x = torch.randn((5, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert len(out) == 3
    assert out[0].shape == (5, 3, 13, 13, num_classes + 5)
    assert out[1].shape == (5, 3, 26, 26, num_classes + 5)
    assert out[2].shape == (5, 3, 52, 52, num_classes + 5)

def test_YOLOLoss():
    anchors = torch.tensor([[0.28, 0.22], [0.38, 0.48], [0.9, 0.78]])
    loss = YOLOLoss()
    num_classes = 2
    predictions = torch.zeros((5, 3, 13, 13, 5 + num_classes))
    targets = torch.zeros((5, 3, 13, 13, 6))
    out = loss(predictions, targets, anchors)
    assert abs(out.item() - 0.693147) < 1e-6 # loss should be close to ln(2)

    predictions[..., 4] = -20   # loss should be close to 0
    out = loss(predictions, targets, anchors)
    assert abs(out.item()) < 1e-6

def main():
    test_CNNBlock()
    test_ResidualBlock()
    test_ScalePredictionBlock()
    test_Yolov3()
    test_YOLOLoss()
    print("All tests passed.")

if __name__ == "__main__":
    main()
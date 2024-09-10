
import torch
from model import CNNBlock, ResidualBlock, ScalePredictionBlock, YOLOv3
from loss import YOLOLoss
from dataset import YOLODataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from utils import create_csv_files, cells_to_boxes, non_max_suppression, plot_image_with_boxes
import config
import os


def test_CNNBlock():
    """
    Make sure output shape of CNNBlock is as expected.
    """
    in_channels = 3
    out_channels = 32
    block = CNNBlock(in_channels, out_channels, kernel_size = 1)
    x = torch.randn((5, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
    out = block(x)
    assert out.shape == (5, 32, config.IMAGE_SIZE, config.IMAGE_SIZE)

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
    x = torch.randn((5, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
    out = model(x)
    assert len(out) == 3
    assert out[0].shape == (5, 3, 13, 13, num_classes + 5)
    assert out[1].shape == (5, 3, 26, 26, num_classes + 5)
    assert out[2].shape == (5, 3, 52, 52, num_classes + 5)

def test_load_weights():
    model = YOLOv3(weights_path = r'C:\Users\tzong\Documents\YOLO-For-Turbines\weights\yolov3.weights')
    model.load_weights()
    num_classes = 80

    x = torch.randn((2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
    out = model(x)
    assert out[0].shape == (2, 3, config.IMAGE_SIZE//32, config.IMAGE_SIZE//32, num_classes + 5)
    assert out[1].shape == (2, 3, config.IMAGE_SIZE//16, config.IMAGE_SIZE//16, num_classes + 5)
    assert out[2].shape == (2, 3, config.IMAGE_SIZE//8, config.IMAGE_SIZE//8, num_classes + 5)

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

def test_YOLOPred():
    """
    Test YOLOv3 model with pretrained weights on sample image from COCO dataset.
    """
    anchors = config.ANCHORS
    model = YOLOv3(weights_path = r'C:\Users\tzong\Documents\YOLO-For-Turbines\weights\yolov3.weights')
    model.load_weights()
    model.eval()

    # print(torch.isnan(model.parameters()).any())
    split_folder = r'C:\Users\tzong\Documents\YOLO-For-Turbines\data'
    img_folder_path = r'C:\Users\tzong\Documents\YOLO-For-Turbines\data\test_images'
    label_folder_path = r'C:\Users\tzong\Documents\YOLO-For-Turbines\data\test_labels'
    create_csv_files(img_folder_path, label_folder_path, split_folder, split_map = {"test_pred": 1})
    dataset = YOLODataset(
        csv_split_file = f"{split_folder}/test_pred.csv",
        img_folder = img_folder_path,
        annotation_folder = label_folder_path,
        anchors = anchors,
        image_size = config.IMAGE_SIZE,
        grid_sizes = config.GRID_SIZES,
        transform = config.test_transforms
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False)
    scaled_anchors = torch.tensor(anchors) * (
                    torch.tensor(config.GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
    )
    x, y = next(iter(loader))

    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = scaled_anchors[i]
            boxes_scale_i = cells_to_boxes(
                out[i], anchor, grid_size=S, is_pred=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=0.5, obj_threshold= 0.7, box_format="midpoint",
        )

        plot_image_with_boxes(x[i].permute(1,2,0).detach().cpu(), nms_boxes, class_list = config.COCO_LABELS)

def main():
    # test_CNNBlock()
    # test_ResidualBlock()
    # test_ScalePredictionBlock()
    # test_Yolov3()
    # test_load_weights()
    # test_YOLOLoss()
    test_YOLOPred()
    print("All tests passed.")

if __name__ == "__main__":
    main()
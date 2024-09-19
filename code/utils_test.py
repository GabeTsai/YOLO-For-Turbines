import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
from model import CNNBlock, ResidualBlock, ScalePredictionBlock, YOLOv3
from loss import YOLOLoss
from dataset import YOLODataset
from PIL import Image

from utils import create_csv_files, cells_to_boxes, non_max_suppression, plot_image_with_boxes, calc_mAP, mosaic_augmentation
import config
import os

def test_iou():
    bbox1 = torch.tensor([0.5, 0.5, 0.25, 0.25])
    bbox2 = torch.tensor([0.5, 0.5, 0.25, 0.25])
    print(calc_iou(bbox1, bbox2))
    assert calc_iou(bbox1, bbox2) == 1.0

def test_mAP():
    pred_boxes = [
        [0, 0.5, 0.5, 0.25, 0.25, 0.9, 0],
        [0, 0.5, 0.5, 0.1, 0.1, 0.6, 0]
    ]
    true_boxes = [
        [0, 0.5, 0.5, 0.25, 0.25, 0.9, 0],
        [0, 0.5, 0.5, 0.1, 0.1, 0.6, 0]
    ]
    print(calc_mAP(pred_boxes, true_boxes))
    assert calc_mAP(pred_boxes, true_boxes) == 1.0

def test_cells_to_boxes():
    num_classes = 3
    grid_size = 3
    predictions = torch.zeros((5, 3, grid_size, grid_size, 5 + num_classes))
    anchors = torch.tensor([[0.28, 0.22], [0.38, 0.48], [0.9, 0.78]])
    boxes = cells_to_boxes(predictions, anchors, grid_size)
    assert torch.tensor(boxes).shape == (5, 3 * grid_size * grid_size, 6)

def test_mosaic():
    image_names = ["DJI_0004_02_05.png", "DJI_0004_02_06.png", "DJI_0004_03_06.png", "DJI_0004_04_05.png"]
    image_paths = [f"{config.IMAGE_FOLDER}/{image_name}" for image_name in image_names]
    label_paths = [f"{config.ANNOTATION_FOLDER}/{image_name[:-4]}.txt" for image_name in image_names]
    images = [np.array(Image.open(img_path).convert('RGB')) for img_path in image_paths]
    print(images[0].shape)
    boxes_list = []
    for i in range(len(label_paths)):
        boxes = np.loadtxt(label_paths[i], delimiter=" ")
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        boxes = np.roll(boxes, shift=4, axis=1)  # Albumentations expects [x, y, w, h, class]
        boxes_list.append(boxes.tolist())

    cutout, new_boxes = mosaic_augmentation(images, boxes_list, size = 416)
    new_boxes = np.insert(new_boxes, 4, 1, axis = 1)
    plot_image_with_boxes(cutout, new_boxes, class_list = config.TURBINE_LABELS)

def main():
    # test_iou()
    # test_mAP()
    # test_cells_to_boxes()
    test_mosaic()
    print("All tests passed.")

if __name__ == "__main__":
    main()
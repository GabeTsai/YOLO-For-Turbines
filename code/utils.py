import pandas as pd
import numpy as np
import torch
from pathlib import Path
import os

def iou_aligned(box1, box2):
    """
    Calculate IoU between two boxes. Assumes boxes are aligned at center.

    Args:
        box1: torch.Tensor, shape (2, ) [w, h]
        box2: torch.Tensor, shape (2, ) [w, h]
    
    Returns:
        torch.Tensor, IoU of boxes.
    """

    intersection = torch.min(box1[..., 0], box2[..., 0]) * torch.min(box1[..., 1], box2[..., 1])
    union = box1[..., 0] * box1[..., 1] + box2[..., 0] * box2[..., 1] - intersection
    return intersection / union

def iou(boxes1, boxes2, mode = "center"):
    """
    Calculate the Intersection over Union (IoU) of arrays of bounding boxes.

    Args:
        boxes1: (N, 4) or (4,) Tensor
        boxes2: (N, 4) or (4,) Tensor
        mode: string specifying "center" for cxcywh or "corner" for x1y1x2y2
    
    Returns:
        (N, ) Tensor
    
    """
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)

    # Convert center coordinates to top left coordinates.
    if mode == "center":
        boxes1[..., :2] = boxes1[..., :2] - boxes1[..., 2:]/2
        boxes2[..., :2] = boxes2[..., :2] - boxes2[..., 2:]/2

    xA = torch.max(boxes1[..., 0], boxes2[..., 0])
    yA = torch.max(boxes1[..., 1], boxes2[..., 1])
    xB = torch.min(boxes1[..., 0] + boxes1[..., 2], boxes2[..., 0] + boxes2[..., 2])
    yB = torch.min(boxes1[..., 1] + boxes1[..., 3], boxes2[..., 1] + boxes2[..., 3])
    box_width = torch.clamp(xB - xA, min=0)
    box_height = torch.clamp(yB - yA, min=0)
    intersection_area = box_width * box_height
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    union_area = boxes1_area + boxes2_area - intersection_area
    iou = intersection_area / union_area
    return iou

def create_csv_files(image_folder, annotation_folder, split_folder, split_map):
    """
    Create csv files for training, validation and test datasets.
    
    Args:
        image_folder: str, path to folder with images
        annotation_folder: str, path to folder with txt files containing annotations
        split_folder: str, path to folder where csv files will be saved
        split_map: dict, keys specify split, values specify percentage
    """
    images = np.array(os.listdir(image_folder))
    labels = np.array(os.listdir(annotation_folder))
    
    image_names = set([image[:-4] for image in images])
    label_names = set([label[:-4] for label in labels])

    assert(len(images) == len(image_names))

    common_names = image_names.intersection(label_names)

    data_list = []

    for image_name in sorted(image_names):
        if image_name in common_names:
            data_list.append([image_name + '.png', image_name + '.txt'])
        else:
            data_list.append([image_name + '.png', None])

    assert(len(data_list) == len(images))
    data_arr = np.array(data_list)

    rng = np.random.default_rng(seed=3407)  
    random_array = rng.integers(len(data_arr), size=len(data_arr))
    data_arr = data_arr[random_array]

    start_idx = 0
    for split in split_map:
        end_idx = start_idx + int(split_map[split] * len(data_arr))
        split_data = data_arr[start_idx:end_idx]
        np.savetxt(Path(f"{split_folder}/{split}.csv"), split_data, fmt = "%s", delimiter = ",")

def seed_everything(seed = 3407):
    """
    Seed all random number generators.
    
    Args:
        seed: int, seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
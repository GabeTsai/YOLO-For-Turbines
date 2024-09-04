"""
YOLO Dataset class. Assumes folder with images, folder with txt file for image annotations,
and csv files for which images are in training, val and test datasets.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path

from PIL import Image, ImageFile
from utils import iou_aligned
from torch.utils.data import Dataset, DataLoader

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_split_file,
        img_folder,
        annotation_folder,
        anchors,
        image_size = 416,
        grid_sizes = [13, 26, 52],
        num_classes = 80,
        transform = None
        ):

        self.annotations = pd.read_csv(csv_split_file, header = None)
        self.img_folder = img_folder
        self.annotation_folder = annotation_folder
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.image_size = image_size
        self.grid_sizes = grid_sizes
        self.num_classes = num_classes
        self.transform = transform
        self.ignore_iou_threshold = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = Path(f"{self.img_folder}/{self.annotations.iloc[idx, 0]}")
        img = np.array(Image.open(img_path).convert('RGB'))
        label_path = Path(f"{self.annotation_folder}/{self.annotations.iloc[idx, 1]}")

        #[3, g, g, objectness, class, tx, ty, tw, th] for g in grid_sizes   
        # Initialize targets to zero; assume no object is present in any anchor by default
        targets = [torch.zeros((self.num_anchors // 3, grid_size, grid_size, 6) for grid_size in self.grid_sizes)]

        if label_path.exists():
            boxes = np.roll(np.loadtxt(label_path, delimiter = " "), shift = 4, axis = 1).tolist() #albumentations expects [x, y, w, h, class]
            if self.transform:
                augmentations = self.transform(image = img, bboxes = boxes)
                img = augmentations['image']
                boxes = augmentations['bboxes']

           
            for box in boxes:
                iou_with_anchors = iou_aligned(torch.tensor(box[2:]), self.anchors)
                anchor_indices = iou_with_anchors.argsort(descending = True, dim = 0)
                class_label, x, y, w, h = box
                #assign an anchor from each scale to the box
                has_anchor = [False] * 3
                for anchor_idx in anchor_indices:
                    scale_idx = anchor_idx // self.num_anchors_per_scale    #find which scale anchor belongs to
                    anchor_for_scale = anchor_idx % self.num_anchors_per_scale  #find specific anchor within scale
                    cur_grid_size = self.grid_sizes[scale_idx]
                    i, j = int(cur_grid_size * x), int(cur_grid_size * y)   #grid cell indices
                    anchor_taken = targets[scale_idx][anchor_for_scale, i, j, 0]

                    if not anchor_taken and not has_anchor[scale_idx]:  #if anchor is free and scale doesn't already have an assigned anchor
                        targets[scale_idx][anchor_for_scale, i, j, 0] = 1
                        x_cell, y_cell = cur_grid_size * x - i, cur_grid_size * y   #get top left coord for specific grid cell
                        width_cell, height_cell = (w * cur_grid_size, h * cur_grid_size)    #scale to grid

                        box_coords = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                        targets[scale_idx][anchor_for_scale, i, j, 1] = int(class_label)
                        targets[scale_idx][anchor_for_scale, i, j, 2:] = box_coords
                        has_anchor[scale_idx] = True

                    #ignore anchors that have significant IoU but aren't the best fit with ground truth
                    #we don't want to train the model on suboptimal anchors.
                    elif not anchor_taken and iou_with_anchors[anchor_idx] > self.ignore_iou_threshold:
                        targets[scale_idx][anchor_for_scale, i, j, 0] = -1  

        return img, tuple(targets)
        
       

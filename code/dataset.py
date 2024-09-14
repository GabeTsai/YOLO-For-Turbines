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
import config
from torch.utils.data import Dataset, DataLoader

import random

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_split_file,
        img_folder,
        annotation_folder,
        anchors,
        batch_size, 
        num_batch_to_resize = 10, 
        image_size = config.DEF_IMAGE_SIZE,
        grid_sizes = [13, 26, 52],
        num_classes = 80,
        transform = None,
        multi_scale = None
        ):

        self.annotations = pd.read_csv(csv_split_file, header = None)
        self.img_folder = img_folder
        self.annotation_folder = annotation_folder
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3

        self.batch_size = batch_size
        self.data_idx = 0
        self.num_batch_to_resize = num_batch_to_resize

        self.image_size = image_size
        self.grid_sizes = grid_sizes
        self.num_classes = num_classes
        self.transform = transform
        self.ignore_iou_threshold = 0.5
        self.multi_scale = multi_scale

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = Path(f"{self.img_folder}/{self.annotations.iloc[idx, 0]}")
        img = np.array(Image.open(img_path).convert('RGB')) 
        label_path = Path(f"{self.annotation_folder}/{self.annotations.iloc[idx, 1]}")
        #[3, g, g, tx + ty + tw + th + objectness + class] for g in grid_sizes   
        # Initialize targets to zero; assume no object is present in any anchor by default
        targets = [torch.zeros((self.num_anchors // 3, grid_size, grid_size, 6)) for grid_size in self.grid_sizes]

        self.data_idx += 1

        if self.multi_scale and (self.data_idx + 1) % (self.batch_size * self.num_batch_to_resize) == 0:
            self.image_size = random.choice(config.MULTI_SCALE_TRAIN_SIZES)
            self.grid_sizes = [self.image_size // 32, self.image_size // 16, self.image_size // 8] 
            self.transform = config.set_train_transforms(self.image_size)

        if label_path.exists():
            boxes = np.loadtxt(label_path, delimiter=" ")
            if boxes.ndim == 1:
                boxes = boxes.reshape(1, -1)
            boxes = np.roll(boxes, shift = 4, axis = 1).tolist()    #albumentations expects [x, y, w, h, class]
            
            if self.transform is not None:
                augmentations = self.transform(image = img, bboxes = boxes)
                img = augmentations['image']
                boxes = augmentations['bboxes']

            for box in boxes:
                iou_with_anchors = iou_aligned(torch.tensor(box[2:4]), self.anchors)
                anchor_indices = iou_with_anchors.argsort(descending = True, dim = 0)
                x, y, w, h, class_label = box
                
                #assign an anchor from each scale to the box
                has_anchor = [False] * 3

                for anchor_idx in anchor_indices:
                    scale_idx = anchor_idx // self.num_anchors_per_scale    #find which scale anchor belongs to
                    anchor_for_scale = anchor_idx % self.num_anchors_per_scale  #find specific anchor within scale
                    cur_grid_size = self.grid_sizes[scale_idx]

                    i, j = int(cur_grid_size * y) , int(cur_grid_size * x)   #grid cell indices
                    anchor_taken = targets[scale_idx][anchor_for_scale, i, j, 0]

                    #if anchor is free and scale doesn't already have an assigned anchor
                    if not anchor_taken and not has_anchor[scale_idx]:  
                        targets[scale_idx][anchor_for_scale, i, j, 4] = 1   #object exists for grid cell
                        x_cell, y_cell = cur_grid_size * x - j, cur_grid_size * y - i   #get top left coord for specific grid cell
                        width_cell, height_cell = (w * cur_grid_size, h * cur_grid_size)    #scale to grid
                        box_coords = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                        assert(torch.allclose(torch.tensor([
                            (x_cell + j) / cur_grid_size,
                            (y_cell + i) / cur_grid_size,
                            width_cell / cur_grid_size,
                            height_cell / cur_grid_size
                        ]), torch.tensor([x, y, w, h]), atol=1e-6))
                        targets[scale_idx][anchor_for_scale, i, j, 5] = int(class_label)
                        targets[scale_idx][anchor_for_scale, i, j, :4] = box_coords
                        has_anchor[scale_idx] = True

                    #ignore anchors that have significant IoU but aren't the best fit with ground truth
                    #we don't want to train the model on suboptimal anchors.
                    elif not anchor_taken and iou_with_anchors[anchor_idx] > self.ignore_iou_threshold:
                        targets[scale_idx][anchor_for_scale, i, j, 4] = -1  

        return img, tuple(targets)
        
       

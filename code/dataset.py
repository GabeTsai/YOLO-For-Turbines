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
from utils import mosaic_augmentation

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
        mosaic = False,
        multi_scale = False
        ):

        self.annotations = pd.read_csv(csv_split_file, header = None)
        self.img_folder = img_folder
        self.annotation_folder = annotation_folder
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3

        self.batch_size = batch_size
        self.num_batch_to_resize = num_batch_to_resize

        self.image_size = image_size
        self.grid_sizes = grid_sizes
        self.num_classes = num_classes
        self.transform = transform
        self.mosaic = mosaic
        self.ignore_iou_threshold = 0.5
        self.multi_scale = multi_scale

    def __len__(self):
        return len(self.annotations)

    def load_image(self, idx):
        """
        Helper function to load an image and its corresponding bounding boxes.
        """
        img_path = Path(f"{self.img_folder}/{self.annotations.iloc[idx, 0]}")
        img = np.array(Image.open(img_path).convert('RGB'), dtype = np.uint8)

        return img

    def load_boxes(self, label_path, idx):
        boxes = np.loadtxt(label_path, delimiter=" ")
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        
        boxes = np.roll(boxes, shift=4, axis=1)  # Albumentations expects [x, y, w, h, class]

        return boxes.tolist()
    
    def apply_augmentations(self, img, boxes, idx):
        """
        Applies augmentations to the image and boxes. Handles both mosaic and non-mosaic cases.
        """
        if self.mosaic:
            # Load three additional images and their labels for mosaic augmentation
            imgs, labels = [img], [boxes]
            for _ in range(3):
                rand_idx = random.randint(0, len(self.annotations) - 1)
                while rand_idx == idx:
                    rand_idx = random.randint(0, len(self.annotations) - 1)
                rand_img = self.load_image(rand_idx)
                rand_label_path = Path(f"{self.annotation_folder}/{self.annotations.iloc[rand_idx, 1]}")
                rand_boxes = self.load_boxes(rand_label_path, rand_idx) if rand_label_path.exists() else []
                imgs.append(rand_img)
                labels.append(rand_boxes)

            # Apply mosaic augmentation
            mosaic, mosaic_boxes = mosaic_augmentation(imgs, labels, self.image_size)

            # If mosaic cutout doesn't contain valid boxes, fallback to regular augmentation
            if mosaic.any() == -1 and mosaic_boxes.any() == -1:
                no_mosaic_transform = config.set_train_transforms(self.image_size, mosaic=False)
                augmentations = no_mosaic_transform(image=img, bboxes=boxes)
            else:
                augmentations = self.transform(image=mosaic, bboxes=mosaic_boxes)

        #We're still in training mode but just use the default train transforms
        elif self.multi_scale:
            standard_transform = config.set_train_transforms(self.image_size, mosaic = False)
            augmentations = standard_transform(image = img, bboxes = boxes)
        #We must be in test mode. Use the dataset's transform.
        else:
            augmentations = self.transform(image=img, bboxes=boxes)

        # Return augmented image and boxes
        return augmentations['image'], augmentations['bboxes']

    def change_scale(self):
        self.image_size = random.choice(config.MULTI_SCALE_TRAIN_SIZES)
        self.grid_sizes = [self.image_size // 32, self.image_size // 16, self.image_size // 8] 
        self.transform = config.set_train_transforms(self.image_size, mosaic = self.mosaic)
        targets = [torch.zeros((self.num_anchors // 3, grid_size, grid_size, 6)) for grid_size in self.grid_sizes]

    def __getitem__(self, idx):
        img = self.load_image(idx)
        #[3, g, g, tx + ty + tw + th + objectness + class] for g in grid_sizes   
        # Initialize targets to zero; assume no object is present in any anchor by default
        targets = [torch.zeros((self.num_anchors // 3, grid_size, grid_size, 6)) for grid_size in self.grid_sizes]
        label_path = Path(f"{self.annotation_folder}/{self.annotations.iloc[idx, 1]}")
        
        if label_path.exists():
            boxes = self.load_boxes(label_path, idx)  
            img, boxes = self.apply_augmentations(img, boxes, idx)
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
                        assert torch.sum(box_coords < 0 ) == 0

                        targets[scale_idx][anchor_for_scale, i, j, 5] = int(class_label)
                        targets[scale_idx][anchor_for_scale, i, j, :4] = box_coords
                        has_anchor[scale_idx] = True

                    #ignore anchors that have significant IoU but aren't the best fit with ground truth
                    #we don't want to train the model on suboptimal anchors.
                    elif not anchor_taken and iou_with_anchors[anchor_idx] > self.ignore_iou_threshold:
                        targets[scale_idx][anchor_for_scale, i, j, 4] = -1  
        else:
            only_image_transform = config.set_only_image_transforms(image_size = self.image_size)
            augmentations = only_image_transform(image = img)
            img = augmentations['image']

        return img, tuple(targets)
        
       

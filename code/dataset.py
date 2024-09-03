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
        self.image_size = image_size
        self.grid_sizes = grid_sizes
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = Path(f"{self.img_folder}/{self.annotations.iloc[idx, 0]}")
        img = np.array(Image.open(img_path).convert('RGB'))
        label_path = Path(f"{self.annotation_folder}/{self.annotations.iloc[idx, 1]}")
        if label_path.exists():
            boxes = np.roll(np.loadtxt(label_path, delimiter = " "), shift = 4, axis = 1).tolist() #albumentations expects [x, y, w, h, class]
            if self.transform:
                augmentations = self.transform(image = img, bboxes = boxes)
                img = augmentations['image']
                boxes = augmentations['bboxes']

            #[3, g, g, tx + ty + tw + th + objectness + class]
            targets = [torch.zeros((self.num_anchors // 3, grid_size, grid_size, 6) for grid_size in self.grid_sizes)]

            for box in boxes:
                iou_with_anchors = iou_aligned(torch.tensor(box[2:]), self.anchors)
                anchor_indices = iou_with_anchors.argsort(descending = True, dim = 0)
                
        else:
            boxes = []
        
       

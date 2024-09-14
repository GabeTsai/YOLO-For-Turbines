import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import torch
import random
import os
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_PROCESSES = 4
NUM_WORKERS = int(os.cpu_count()/NUM_PROCESSES) if int(os.cpu_count()/NUM_PROCESSES) <  16 else 16
NUM_GPUS = torch.cuda.device_count()
PIN_MEMORY = True

MAP_IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.3
NMS_IOU_THRESHOLD = 0.45

IMAGE_FOLDER = "../data/images"
ANNOTATION_FOLDER = "../data/labels"
WEIGHTS_FOLDER = "../weights"
WEIGHTS = Path(f"{WEIGHTS_FOLDER}/yolov3.weights")
DEF_IMAGE_SIZE = 416

MULTI_SCALE_TRAIN_SIZES = [
    320, 352, 384, 416, 448, 480, 512, 544, 576, 608
]

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
] 
GRID_SIZES = [DEF_IMAGE_SIZE//32, DEF_IMAGE_SIZE//16, DEF_IMAGE_SIZE//8]

def set_train_transforms(image_size = DEF_IMAGE_SIZE):
    scale = random.uniform(1.0, 1.5)

    train_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=int(image_size * scale)), # resize so we can crop and shift image
            A.PadIfNeeded(
                min_height=int(image_size * scale),
                min_width=int(image_size * scale),
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.RandomCrop(width=image_size, height=image_size),
            A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
            A.OneOf(    
                [
                    A.ShiftScaleRotate(
                        rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                    ),
                    A.IAAAffine(shear=15, p=0.5, mode="constant"),
                ],
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.Blur(p=0.1),
            A.CLAHE(p=0.1),
            A.Posterize(p=0.1),
            A.ToGray(p=0.1),
            A.ChannelShuffle(p=0.05),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
    )
    return train_transforms

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=DEF_IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=DEF_IMAGE_SIZE, min_width=DEF_IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value = 255
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

TURBINE_LABELS = ["dirt", "damage"]
NUM_TURBINE_CLASSES = len(TURBINE_LABELS)

COCO_LABELS = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]
NUM_COCO_CLASSES = len(COCO_LABELS)
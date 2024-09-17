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
CONF_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.45

PROJ_FOLDER = "/home/groups/yzwang/gabriel_files/YOLO-For-Turbines"
IMAGE_FOLDER = f"{PROJ_FOLDER}/data/images"
ANNOTATION_FOLDER = f"{PROJ_FOLDER}/data/labels"
WEIGHTS_FOLDER = f"{PROJ_FOLDER}/weights"
MODEL_FOLDER = f"{PROJ_FOLDER}/models"
CSV_FOLDER = f"{PROJ_FOLDER}/data"

COCO_WEIGHTS = Path(f"{WEIGHTS_FOLDER}/yolov3.weights")
DARKNET_WEIGHTS = Path(WEIGHTS_FOLDER) / "darknet53.conv.74"
FREEZE_BACKBONE = True
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

    train_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=int(image_size)), # resize image
            A.PadIfNeeded(
                min_height=int(image_size),
                min_width=int(image_size),
                border_mode=cv2.BORDER_CONSTANT, 
                value = 255
            ),
            A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
            A.OneOf(    
                [
                    A.ShiftScaleRotate(
                        rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT, value = 255
                    ),
                    A.Affine(shear=15, mode=0, p=0.5),
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
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[], clip = True),
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
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[], clip = True),
)

def set_only_image_transforms(image_size = DEF_IMAGE_SIZE):
    only_image_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size = image_size),
            A.PadIfNeeded(
                min_height= image_size, min_width= image_size, border_mode=cv2.BORDER_CONSTANT, value = 255
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
            ToTensorV2(),
        ]
    )

    return only_image_transforms


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
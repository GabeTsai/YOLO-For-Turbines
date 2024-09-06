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

def calc_iou(boxes1, boxes2, box_format = "center"):
    """
    Calculate the Intersection over Union (IoU) of arrays of bounding boxes.

    Args:
        boxes1: (N, 4) or (4,) Tensor
        boxes2: (N, 4) or (4,) Tensor
        box_format: string specifying "center" for cxcywh or "corner" for x1y1x2y2
    
    Returns:
        (N, ) Tensor
    
    """
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)

    # Convert center coordinates to top left coordinates.
    if box_format == "center":
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

def cell_pred_to_boxes(predictions, anchors, grid_size, is_pred = True):
    """
    Convert YOLO cell predictions for one scale to bounding box predictions in cxcywh form.

    Args:
        predictions: torch.Tensor, shape (N, 3, S, S, 5 + num_classes) or (N, 3, S, S, 5 + class_label)
        anchors: torch.Tensor, shape (3, 2), anchors for the scale
        grid_size: int, size of the grid
        is_pred: bool, whether predictions are in raw form or sigmoid form

    Returns:
        List of lists, each list is bounding box in format [cx, cy, w, h, obj, class_label]
    """
    
    batch_size = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., :4]

    if is_pred:
        #Convert to absolute coordinates relative to grid cell
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        #Reshape anchors for broadcasting operations
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2) #(1, 3, 1, 1, 2)
        #Convert to actual width and height in pixels
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        object_scores = torch.sigmoid(predictions[..., 4:5])
        best_class = torch.argmax(predictions[..., 5:], dim = -1).unsqueeze(-1)
    
    else:
        object_scores = predictions[..., 0:1]
        best_class = predictions[..., 5:]

    #We create grid cell indices to adjust box coordinates to be relative to entire image
    grid_cell_indices = (
        torch.arange(grid_size)
        .repeat(batch_size, 3, grid_size, 1)    #(B, 3, grid_size, grid_size)
        .unsqueeze(-1)  #(B, 3, grid_size, grid_size, 1)
        .to(predictions.device)
    )
    cx = 1 / grid_size * (box_predictions[..., 1:2] + grid_cell_indices)

    #With permute (0, 1, 3, 2, 4), 
    #We effectively transpose the grid indices, or dimensions 2 and 3
    #let's say our grid is a 3 by 3
    # [[0, 1, 2]
    #  [0, 1, 2]
    #  [0, 1, 2]]
    # becomes 
    # [[0, 0, 0],
    #  [1, 1, 1],, 
    #  [2, 2, 2]]
    #When we combine that with the original grid indices, we get the grid coordinates (top left corner of each grid)
    #[(0, 0), (1, 0), (2, 0)
    # (0, 1), (1, 1), (2, 1),
    # (0, 2), (1, 2), (2, 2)]

    cy = 1 / grid_size * (box_predictions[..., 1:2] + grid_cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / grid_size * (box_predictions[..., 2:])
    converted_boxes = torch.cat((cx, cy, w_h, object_scores, best_class), dim = -1)
    #(B, 3, S, S, 6) --> #(B, 3*S*S, 6)
    converted_boxes = converted_boxes.reshape(batch_size, num_anchors * grid_size * grid_size, 6)
    return converted_boxes.tolist()

def non_max_suppression(boxes, iou_threshold, obj_threshold, box_format = "corners"):
    """
    Performs Non Max Suppression given a set of boxes.

    Args:
        boxes: list of lists, each list is bounding box in format [x, y, w, h, obj, class_label]
        iou_threshold: float, IoU threshold for overlapping boxes
        obj_threshold: float, objectness threshold for filtering out boxes before any NMS
        box_format: string, format of the boxes, either "corners" or "center"

    Returns:
        list of boxes given a specific IoU threshold. 
        boxes are sorted by objectness score in descending order, format [x, y, w, h, obj, class_label]
    """
    
    filtered_boxes = [box for box in boxes if box[4] > obj_threshold]
    filtered_boxes = sorted(filtered_boxes, key = lambda x: x[4], reverse = True)
    nms_boxes = []

    while filtered_boxes:
        largest_score_box = filtered_boxes.pop(0)
        nms_boxes.append(largest_score_box)
        filtered_boxes = [
            box for box in filtered_boxes
            if box[4] != largest_score_box[4]
            or calc_iou(torch.tensor(largest_score_box[:4]), torch.tensor(box[:4]), 
                   box_format = box_format) < iou_threshold
        ]

    return nms_boxes

def calc_mAP(pred_boxes, true_boxes, iou_threshold = 0.5, box_format = "center", num_classes = 20):
    """
    Calculate mean Average Precision (mAP) for object detection.

    Args:
        pred_boxes: list of lists, each list is bounding box in format [id, x, y, w, h, obj, class_label]
        true_boxes: list of lists, each list is bounding box in format [id, x, y, w, h, obj, class_label]
        iou_threshold: float, IoU threshold for overlapping boxes

    Returns:
        float, mAP across all classes given IoU threshold
    """

    average_precisions = []

    for c in range(num_classes):
        #Select boxes that belong to class c
        detections = [detection for detection in pred_boxes if detection[-1] == c]
        ground_truths = [true_box for true_box in true_boxes if true_box[-1] == c]

        total_true_boxes = len(ground_truths)

        #Image has no boxes for that class. Skip to next class
        if total_true_boxes == 0:
            continue

        bboxes_per_image = {}

        #Generate a dictionary {image_id: torch.zeros(number of ground truth boxes for that image)}
        for truth in ground_truths:
            bboxes_per_image[truth[0]] = bboxes_per_image.get(truth[0], 0) + 1
        
        for image_id, num_boxes in bboxes_per_image.items():
            bboxes_per_image[image_id] = torch.zeros(num_boxes)
        
        #Sort by descending objectness score
        detections.sort(key = lambda x: x[5], reverse = True)

        #Prepare true positives and false positives for AP calculation
        TP, FP = torch.zeros(len(detections)), torch.zeros(len(detections))

        for detection_idx, detection in enumerate(detections):
            #Get all ground truths that have same image as detection
            ground_truth_boxes_img = [
                true_box for true_box in ground_truths if true_box[0] == detection[0]
            ]

            best_iou = 0
            best_ground_truth_idx = 0   #index of ground truth box with best iou with detection
            
            #Calculate the IoU of the detection with all ground truth boxes
            for truth_idx, ground_truth in enumerate(ground_truth_boxes_img):
                iou = calc_iou(torch.tensor(detection[1:5]), 
                               torch.tensor(ground_truth[1:5]), box_format = box_format)
                if iou > best_iou:
                    best_iou = iou
                    best_ground_truth_idx = truth_idx

            #We meet the iou criteria, but check if the ground truth box has already been assigned to another detection
            if best_iou > iou_threshold:
                if bboxes_per_image[detection[0]][best_ground_truth_idx] == 0:  #if ground truth box hasn't been assigned
                    TP[detection_idx] = 1
                    bboxes_per_image[detection[0]][best_ground_truth_idx] = 1
                else:   #we've already assigned this ground truth box to another detection
                    FP[detection_idx] = 1
            #We don't meet the iou criteria - FP
            else:
                FP[detection_idx] = 1
        
        cum_TP = torch.cumsum(TP, dim = 0)
        cum_FP = torch.cumsum(FP, dim = 0)
        precisions = cum_TP/(cum_TP + cum_FP)
        recalls = cum_TP/total_true_boxes 

        #When recall is 0, precision is 1 (no false positives yet)
        #We add (0, 1) for numerical integration
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        #Get the area under the precision-recall curve
        average_precisions.append(torch.trapz(precisions, recalls))
    
    return sum(average_precisions) / len(average_precisions)

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
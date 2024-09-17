import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os

import config

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

    # Convert center coordinates to top left coordinates without modifying in place
    if box_format == "center":
        boxes1_converted = torch.zeros_like(boxes1)
        boxes2_converted = torch.zeros_like(boxes2)
        boxes1_converted[..., :2] = boxes1[..., :2] - boxes1[..., 2:] / 2
        boxes1_converted[..., 2:4] = boxes1[..., 2:4]

        boxes2_converted[..., :2] = boxes2[..., :2] - boxes2[..., 2:] / 2
        boxes2_converted[..., 2:4] = boxes2[..., 2:4]
    else:
        boxes1_converted = boxes1
        boxes2_converted = boxes2

    # Now compute IoU using the converted boxes
    xA = torch.max(boxes1_converted[..., 0], boxes2_converted[..., 0])
    yA = torch.max(boxes1_converted[..., 1], boxes2_converted[..., 1])
    xB = torch.min(boxes1_converted[..., 0] + boxes1_converted[..., 2], boxes2_converted[..., 0] + boxes2_converted[..., 2])
    yB = torch.min(boxes1_converted[..., 1] + boxes1_converted[..., 3], boxes2_converted[..., 1] + boxes2_converted[..., 3])
    
    box_width = torch.clamp(xB - xA, min=0)
    box_height = torch.clamp(yB - yA, min=0)
    intersection_area = box_width * box_height
    
    boxes1_area = boxes1_converted[..., 2] * boxes1_converted[..., 3]
    boxes2_area = boxes2_converted[..., 2] * boxes2_converted[..., 3]
    union_area = boxes1_area + boxes2_area - intersection_area
    
    iou = intersection_area / (union_area + 1e-6)
    return iou

def cells_to_boxes(predictions, anchors, grid_size, is_pred = True):
    """
    Convert YOLO cell predictions or targets for one scale to bounding box predictions in cxcywh form.

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
        #Convert to normalized coordinates relative to grid cell
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        #Reshape anchors for broadcasting operations
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2) #(1, 3, 1, 1, 2)
        #Convert to actual width and height in pixels
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        object_scores = torch.sigmoid(predictions[..., 4:5])
        best_class = torch.argmax(predictions[..., 5:], dim = -1).unsqueeze(-1)
    
    else:
        object_scores = predictions[..., 4:5]
        best_class = predictions[..., 5:]

    #We create grid cell indices to adjust box coordinates to be relative to entire image
    grid_cell_indices = (
        torch.arange(grid_size)
        .repeat(batch_size, 3, grid_size, 1)    #(B, 3, grid_size, grid_size)
        .unsqueeze(-1)  #(B, 3, grid_size, grid_size, 1)
        .to(predictions.device)
    )
    cx = 1 / grid_size * (box_predictions[..., 0:1] + grid_cell_indices)

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
    filtered_boxes = torch.tensor(sorted(filtered_boxes, key = lambda x: x[4], reverse = True))
    
    nms_boxes = []
    
    while filtered_boxes.size(0) > 0:
        
        largest_score_box = filtered_boxes[0]

        filtered_boxes = filtered_boxes[1:]
        ious = calc_iou(largest_score_box[:4].unsqueeze(0), filtered_boxes[:, :4], box_format)

        # Keep boxes with different class labels or IoU less than the threshold
        class_mask = filtered_boxes[:, 5] != largest_score_box[5]  # Different class
        iou_mask = ious < iou_threshold  # IoU less than threshold

        # Apply both masks (keep boxes either with different classes or low IoU)
        mask = class_mask | iou_mask

        # Filter the boxes
        filtered_boxes = filtered_boxes[mask]
        
        nms_boxes.append(largest_score_box)
    
    nms_boxes = torch.stack(nms_boxes).tolist() if len(nms_boxes) > 0 else []
    
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

def get_eval_boxes(predictions, targets, iou_threshold, anchors, obj_threshold, box_format = "center", device = config.DEVICE):
    """
    Return bounding box predictions and true boxes for evaluation.

    Args:
        predictions: list of lists containing 3 tensors, each tensor is shape (N, 3, S, S, 5 + num_classes)
        targets: list of lists containing 3 tensors, each tensor is shape (N, 3, S, S, 6)
        iou_threshold: float, IoU threshold for overlapping boxes
        anchors: tensor, shape (3, 3, 2), anchors for each scale
        obj_threshold: float, objectness threshold for filtering out boxes before any NMS
        box_format: string, format of the boxes, either "corners" or "center"
    
    """
    data_idx = 0
    all_box_predictions = []
    all_true_boxes = []

    for batch_idx, batch_prediction in enumerate(tqdm(predictions)):
        batch_size = batch_prediction[0].shape[0]
        batch_boxes = [[] for _ in range(batch_size)]
        for i in range(3):  #for each scale
            grid_size = batch_prediction[i].shape[2]  
            #Get anchors in 3 by 2 shape
            anchors_for_scale = torch.tensor([*anchors[i]]).to(device) * grid_size
            boxes_scale_i = cells_to_boxes(
                batch_prediction[i], anchors_for_scale, grid_size, is_pred = True
            )
            #Add box predictions for each image for this particular scale
            for idx, (box) in enumerate(boxes_scale_i):
                batch_boxes[idx] += box

        #Every target box is assigned an anchor from each scale
        #So we can use the last scale from the previous for loop to get the true boxes
        batch_true_boxes = cells_to_boxes(
            targets[batch_idx][2], anchors_for_scale, grid_size, is_pred = False
        )

        for i in range(batch_size):
            nms_boxes = non_max_suppression(
                batch_boxes[i], iou_threshold = iou_threshold, 
                obj_threshold = obj_threshold, box_format = box_format
            )
            #add image id to each box so we know which image it belongs to
            for nms_box in nms_boxes:   
                all_box_predictions.append([data_idx] + nms_box)

            for batch_true_box in batch_true_boxes[i]:
                if batch_true_box[4] > obj_threshold:
                    all_true_boxes.append([data_idx] + batch_true_box)

            data_idx += 1
    return all_box_predictions, all_true_boxes

def check_model_accuracy(predictions, targets, object_threshold):
    """
    Calculate the class, no object and object accuracy of the predicted class labels.

    Args:
        predictions: list of lists containing 3 tensors, each tensor is shape (N, 3, S, S, 5 + num_classes)
        targets: list of lists containing 3 tensors, each tensor is shape (N, 3, S, S, 6)
        object_threshold: float, objectness (confidence) threshold for filtering out boxes before any NMS

    """

    total_class_preds, num_correct_class = 0, 0
    total_noobj, num_correct_noobj = 0, 0
    total_obj, num_correct_obj = 0, 0

    #for each image
    for prediction, target in zip(predictions, targets):
        #for each scale
        for i in range(len(prediction)):
            target[i] = target[i].to(config.DEVICE)
            obj_mask = target[i][..., 4] == 1
            noobj_mask = target[i][..., 4] == 0

            num_correct_class += torch.sum(
                torch.argmax(prediction[i][..., 5:][obj_mask], dim = -1) == target[i][..., 5][obj_mask]
            )
            total_class_preds += torch.sum(obj_mask)

            object_preds = torch.sigmoid(prediction[i][..., 4]) > object_threshold
            num_correct_obj += torch.sum(object_preds[obj_mask] == target[i][..., 4][obj_mask])
            total_obj += torch.sum(obj_mask)

            num_correct_noobj += torch.sum(object_preds[noobj_mask] == target[i][..., 4][noobj_mask])
            total_noobj += torch.sum(noobj_mask)

    class_accuracy = num_correct_class/(total_class_preds + 1e-16)
    noobj_accuracy = num_correct_noobj/(total_noobj + 1e-16)
    obj_accuracy = num_correct_obj/(total_obj + 1e-16)

    print(f"Class accuracy is: {(class_accuracy)*100:2f}%")
    print(f"No obj accuracy is: {(noobj_accuracy)*100:2f}%")
    print(f"Obj accuracy is: {(obj_accuracy)*100:2f}%")
    
    return class_accuracy, noobj_accuracy, obj_accuracy
    
def save_checkpoint(model, optimizer, filename = "YOLOv3TurbineCheckpoint.pth.tar"):
    """
    Save model and optimizer state to checkpoint file.

    Args:
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
        filename: str, name of checkpoint file
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, lr, filename = "YOLOv3TurbineCheckpoint.pth.tar"):
    """
    
    Load model and optimizer state from checkpoint file.
    
    Args:
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
        lr: float, learning rate
        filename: str, name of checkpoint file
    """

    checkpoint = torch.load(filename, map_location = config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def plot_image_with_boxes(image, boxes, class_list):
    """
    Plot image with bounding boxes.

    Args:
        image: np.array, shape (H, W, C)
        boxes: list of lists, each list is bounding box in format [x, y, w, h, obj, class_label]
        class_list: list of strings, class labels
    """
    
    if (len(boxes) == 0):
        print("No objects detected.")
        return
    fig, ax = plt.subplots()
    image = np.array(image)

    ax.imshow(image)
    
    im_h, im_w = image.shape[0], image.shape[1]
    print(im_h, im_w)
    for box in boxes:
        x, y, w, h, _, class_label = box
        top_left_x = (box[0] - box[2]/2) * im_w
        top_left_y = (box[1] - box[3]/2) * im_h
        box_w = w * im_w
        box_h = h * im_h
        rect = patches.Rectangle((top_left_x, top_left_y), box_w, box_h, linewidth = 1, edgecolor = 'r', facecolor = 'none')
        ax.add_patch(rect)

        class_label = int(class_label)
        class_name = class_list[class_label]
        plt.text(top_left_x, top_left_y, s = class_name, 
                 size = 'xx-small',
                 color = 'white', 
                 bbox={"color": "red", "pad": 0})

    plt.axis('off')
    plt.show()
    plt.savefig("example.png", bbox_inches='tight', pad_inches=0)

def plot_original(original_image, resized_image, boxes, class_list):
    o_h, o_w, _ = original_image.shape
    r_h, r_w, _ = resized_image.shape

    # Calculate scaling factor based on the original size and the resized image
    scale = min(r_w / o_w, r_h / o_h)
    
    # Compute the new dimensions of the resized image (before padding)
    new_width = int(o_w * scale)
    new_height = int(o_h * scale)

    # Calculate padding applied to the resized image
    pad_width = (r_w - new_width) // 2
    pad_height = (r_h - new_height) // 2
    
    adjusted_boxes = []
    for box in boxes:
        # Remove padding and convert normalized coordinates to pixel values in the original image
        x_center = (box[0] * r_w - pad_width) / new_width
        y_center = (box[1] * r_h - pad_height) / new_height
        width = (box[2] * r_w) / new_width
        height = (box[3] * r_h) / new_height

        adjusted_boxes.append([x_center, y_center, width, height, box[4], box[5]])
    # Now plot the boxes on the original image
    plot_image_with_boxes(original_image, adjusted_boxes, class_list)
    
def collate_fn(batch):
    """
    Custom collate function to handle different image sizes in a batch.
    
    Args:
        batch: List of tuples (image, target) where `image` is a numpy array 
        (H, W, C) and `target` is the corresponding label or target.

    Returns:
        padded_images: A tensor of padded images, all with the same size.
        targets: A list of the corresponding targets for the images.
    """

    images, targets = zip(*batch)

    max_height = max([img.shape[1] for img in images])  # Max height
    max_width = max([img.shape[2] for img in images])   # Max width

    pad_transform = A.Compose([
        A.PadIfNeeded(min_height=max_height, min_width=max_width, border_mode=cv2.BORDER_CONSTANT, value = 255),  # Padding to max height/width
        ToTensorV2()  
    ])

    padded_images = []
    batched_targets = []

    for img, target in zip(images, targets):        
        augmented = pad_transform(image=np.array(img.permute(1, 2, 0)))
        img_padded = augmented['image']  

        padded_images.append(img_padded)

        batched_targets.append(target)

    padded_images = torch.stack(padded_images)
    #transpose batched_targets so rows are tuples containing all targets of single scale for each image
    #then, stack those rows to get list of three elements where each element is tensor [B, 3, g, g, 6]
    batched_targets = [torch.stack(targets) for targets in zip(*batched_targets)]
    return padded_images, batched_targets

def get_loaders(csv_folder_path, batch_size):
    """
    Get DataLoader objects for training and testing datasets.

    Args:
        train_csv_path: str, path to csv file with training data
        batch_size: int

    Returns:
        DataLoader objects for training and testing datasets
    """
    from dataset import YOLODataset

    IMAGE_SIZE = config.DEF_IMAGE_SIZE
    train_csv_path = Path(csv_folder_path) / "train.csv"
    val_csv_path = Path(csv_folder_path) / "val.csv"
    test_csv_path = Path(csv_folder_path) / "test.csv"

    train_dataset = YOLODataset(csv_split_file = train_csv_path, 
        img_folder = config.IMAGE_FOLDER,
        annotation_folder = config.ANNOTATION_FOLDER,
        anchors = config.ANCHORS,
        batch_size = batch_size,
        image_size = IMAGE_SIZE,
        grid_sizes = config.GRID_SIZES,
        num_classes = config.NUM_TURBINE_CLASSES,
        transform = config.set_train_transforms(image_size = IMAGE_SIZE),
        multi_scale = True
        )
    
    val_dataset = YOLODataset(csv_split_file = val_csv_path,
        img_folder = config.IMAGE_FOLDER,
        annotation_folder = config.ANNOTATION_FOLDER,
        anchors = config.ANCHORS,
        batch_size = batch_size,
        image_size = IMAGE_SIZE,
        grid_sizes = config.GRID_SIZES,
        num_classes = config.NUM_TURBINE_CLASSES,
        transform = config.test_transforms
        )
    
    test_dataset = YOLODataset(csv_split_file = test_csv_path,
        img_folder = config.IMAGE_FOLDER,
        annotation_folder = config.ANNOTATION_FOLDER,
        anchors = config.ANCHORS,
        batch_size = batch_size,
        image_size = IMAGE_SIZE,
        grid_sizes = config.GRID_SIZES,
        num_classes = config.NUM_TURBINE_CLASSES,
        transform = config.test_transforms
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = config.NUM_WORKERS,
        pin_memory = config.PIN_MEMORY,
        # collate_fn = collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = config.NUM_WORKERS,
        pin_memory = config.PIN_MEMORY, 
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = config.NUM_WORKERS,
        pin_memory = config.PIN_MEMORY, 
    )

    return train_loader, val_loader, test_loader
   
def create_csv_files(image_folder, annotation_folder, split_folder, split_map):
    """
    Create csv files for training, validation and test datasets.
    
    Args:
        image_folder: str, path to folder with images
        annotation_folder: str, path to folder with txt files containing annotations
        split_map: dict, keys specify split, values specify percentage
    """
    images = np.array(os.listdir(image_folder))
    labels = np.array(os.listdir(annotation_folder))
    
    image_names = set([image[:-4] for image in images])
    label_names = set([label[:-4] for label in labels])

    assert(len(images) == len(image_names))

    common_names = image_names.intersection(label_names)
    
    data_list = []

    has_obj_count = len(common_names)
    negative_count = 0
    for image_name in sorted(image_names):
        if image_name in common_names:
            data_list.append([image_name + '.png', image_name + '.txt'])
        elif negative_count < has_obj_count:
            data_list.append([image_name + '.png', None])
            negative_count += 1
    
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

if __name__ == "__main__":
    create_csv_files(config.IMAGE_FOLDER, config.ANNOTATION_FOLDER, config.CSV_FOLDER, {"train": 0.70, "val": 0.20, "test": 0.10 })
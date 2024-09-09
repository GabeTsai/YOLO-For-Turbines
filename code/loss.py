import torch
import torch.nn as nn

from utils import calc_iou

class YOLOLoss(nn.Module):
    """
    Compute the YOLO loss function for a single scale.

    Attributes:
        lambda_box: float, weight for box coordinates loss
        lambda_obj: float, weight for objectness loss
        lambda_noobj: float, weight for no objectness loss
        lambda_class: float, weight for class loss

    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lambda_box = 1
        self.lambda_obj = 1
        self.lambda_noobj = 1
        self.lambda_class = 1
    
    def forward(self, predictions, targets, anchors):
        """
        Compute the YOLO loss function for a single scale.

        Args:
            Predictions: torch.Tensor, shape (N, 3, S, S, 4 + objectness + num_classes)
            Targets: torch.Tensor, shape (N, 3, S, S, 4 + objectness + class_label)
            Anchors: torch.Tensor, shape (3, 2)

        Returns:
            float, total loss
        """

        obj_mask = targets[..., 4] == 1
        no_obj_mask = targets[..., 4] == 0

        anchors = anchors.reshape(1, 3, 1, 1, 2)

        # Initialize losses
        no_obj_loss = torch.tensor(0.0, device=predictions.device)
        object_loss = torch.tensor(0.0, device=predictions.device)
        box_loss = torch.tensor(0.0, device=predictions.device)
        class_loss = torch.tensor(0.0, device=predictions.device)

        #NO OBJECT LOSS - penality for predicting object in no object cell
        no_obj_loss = self.bce_logits(predictions[..., 4][no_obj_mask], targets[..., 4][no_obj_mask])
        
        #If object is present in cell
        if obj_mask.any():
            #OBJECT LOSS - how much of the object do we actually capture in the box
            #Compute box centroid relative to grid cell and width and height of box
            box_centroid_pred = self.sigmoid(predictions[..., :2]) # (N, 3, S, S, 2)
            box_wh_pred = torch.exp(predictions[..., 2:4]) * anchors # (N, 3, S, S, 2)
            
            box_preds = torch.cat([box_centroid_pred, box_wh_pred], dim = -1) # (N, 3, S, S, 4)
            iou_preds = iou(box_preds[obj_mask], targets[..., :4][obj_mask]).detach() # (N, 3, S, S)
            object_loss = self.mse((predictions[..., 0:1][obj_mask]), (iou_preds * targets[..., 4][obj_mask]))

            #BOX LOSS - how well do we predict the box coordinates
            #Convert to width and height to offsets for numerical stability
            predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3]) # (N, 3, S, S, 2)
            targets[..., 2:4] = torch.log((1e-16 + targets[..., 2:4] / anchors)) #Convert width and height back to offsets
            box_loss = self.mse(predictions[..., :4][obj_mask], targets[..., :4][obj_mask])

            #CLASS LOSS - how well do we predict the class label
            class_loss = self.cross_entropy(predictions[..., 5:][obj_mask], targets[..., 5][obj_mask])
        
        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_obj_loss
            + self.lambda_class * class_loss
        )


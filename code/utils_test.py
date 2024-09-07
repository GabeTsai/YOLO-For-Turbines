from utils import calc_iou, calc_mAP, cell_pred_to_boxes
import torch

def test_iou():
    bbox1 = torch.tensor([0.5, 0.5, 0.25, 0.25])
    bbox2 = torch.tensor([0.5, 0.5, 0.25, 0.25])
    print(calc_iou(bbox1, bbox2))
    assert calc_iou(bbox1, bbox2) == 1.0

def test_mAP():
    pred_boxes = [
        [0, 0.5, 0.5, 0.25, 0.25, 0.9, 0],
        [0, 0.5, 0.5, 0.1, 0.1, 0.6, 0]
    ]
    true_boxes = [
        [0, 0.5, 0.5, 0.25, 0.25, 0.9, 0],
        [0, 0.5, 0.5, 0.1, 0.1, 0.6, 0]
    ]
    print(calc_mAP(pred_boxes, true_boxes))
    assert calc_mAP(pred_boxes, true_boxes) == 1.0

def test_cell_pred_to_boxes():
    num_classes = 3
    grid_size = 3
    predictions = torch.zeros((5, 3, grid_size, grid_size, 5 + num_classes))
    anchors = torch.tensor([[0.28, 0.22], [0.38, 0.48], [0.9, 0.78]])
    boxes = cell_pred_to_boxes(predictions, anchors, grid_size)
    assert torch.tensor(boxes).shape == (5, 3 * grid_size * grid_size, 6)

def main():
    test_iou()
    test_mAP()
    test_cell_pred_to_boxes()
    print("All tests passed.")

if __name__ == "__main__":
    main()
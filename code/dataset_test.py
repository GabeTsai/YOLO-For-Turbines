from utils import *
from dataset import *
from config import *

def test_create_csv_files():
    image_folder = "data/images"
    annotation_folder = "data/labels"
    split_folder = "data"
    split_map = {"train": 0.8, "val": 0.1, "test": 0.1}
    split_data_map = create_csv_files(image_folder, annotation_folder, split_folder, split_map)

def test_YOLODataset():
    anchors_norm = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ] 
    grid_sizes=[13, 26, 52]

    train_loader, val_loader, _ = get_loaders(config.CSV_FOLDER, batch_size = 10)

    scaled_anchors = torch.tensor(anchors_norm) * (
                    torch.tensor(grid_sizes).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
    )
    # seed_everything()
    for x, y in train_loader:
        boxes = []
        for i in range(y[0].shape[1]): # for each anchor:
            anchor = scaled_anchors[i]
            boxes += cells_to_boxes(y[i], anchors = anchor, grid_size = y[i].shape[2], is_pred = False)[0]
        boxes = non_max_suppression(boxes, iou_threshold = 0.5, obj_threshold = 0.7, box_format = "midpoint")
        print(x.shape)
        # if len(boxes) > 0:
        #     plot_image_with_boxes(x[0].permute(1,2,0), boxes, class_list = TURBINE_LABELS, image_name = "dataset_ex")
        #     break

def main():
    # test_create_csv_files()
    test_YOLODataset()
    print("All tests passed.")

if __name__ == "__main__":
    main()
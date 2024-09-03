from utils import *
from dataset import *

def test_create_csv_files():
    image_folder = "data/images"
    annotation_folder = "data/labels"
    split_folder = "data"
    split_map = {"train": 0.8, "val": 0.1, "test": 0.1}
    split_data_map = create_csv_files(image_folder, annotation_folder, split_folder, split_map)

def test_YOLODataset():
    ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ] 
    dataset = YOLODataset(
        "data/train.csv",
        "data/images/",
        "data/labels/",
        grid_sizes=[13, 26, 52],
        anchors=ANCHORS,
        num_classes = 2,
        transform=None,
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    dataiter = iter(loader)
    images, labels = next(dataiter)
    print(images, labels)

def main():
    test_create_csv_files()
    test_YOLODataset()
    print("All tests passed.")

if __name__ == "__main__":
    main()
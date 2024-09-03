from utils import *

def test_create_csv_files():
    image_folder = "data/images"
    annotation_folder = "data/labels"
    split_folder = "data"
    split_map = {"train": 0.8, "val": 0.1, "test": 0.1}
    split_data_map = create_csv_files(image_folder, annotation_folder, split_folder, split_map)

def main():
    test_create_csv_files()
    print("All tests passed.")

if __name__ == "__main__":
    main()
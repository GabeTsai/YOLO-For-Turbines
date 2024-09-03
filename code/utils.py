import pandas as pd
import numpy as np
import torch
from pathlib import Path
import os

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

    
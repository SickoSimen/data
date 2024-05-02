import os
import numpy as np
from monai.data import CacheDataset
import re


root_dir = "/datasets/tdt4265/mic/asoca"


def get_train_and_val_ds(train_transforms=None, val_transforms=None, validation_fraction = 0.1):
    image_paths = []
    annotation_paths = []


    # Store image and annotation paths
    for subdir in ["Normal", "Diseased"]:
        image_dir = os.path.join(root_dir, subdir, "CTCA")
        annotation_dir = os.path.join(root_dir, subdir, "Annotations")
        image_paths.extend([os.path.join(image_dir, filename) for filename in os.listdir(image_dir)])
        annotation_paths.extend([os.path.join(annotation_dir, filename) for filename in os.listdir(annotation_dir)])

    image_paths = sorted(image_paths, key=extract_number_and_type)
    annotation_paths = sorted(annotation_paths, key=extract_number_and_type)

    for image_path, annotation_path in zip(image_paths, annotation_paths):
        image_name = image_path.split("/")[-1]
        annotation_name = annotation_path.split("/")[-1]
        print(f"Image name: {image_name}, Annotation name: {annotation_name}")


    training_data = [{"image": image_path, "label": annotation_path} for image_path, annotation_path in zip(image_paths, annotation_paths)]


    indices = list(range(len(image_paths)))
    split_idx = int(np.floor(validation_fraction * len(image_paths)))

    val_indices = np.random.choice(indices, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_data = [training_data[i] for i in train_indices]
    val_data = [training_data[i] for i in val_indices]

    train_ds = CacheDataset(data=train_data, transform=train_transforms)
    val_ds = CacheDataset(data=val_data, transform=val_transforms)

    return train_ds, val_ds

def get_test_ds(test_transforms):
    test_image_paths = []

    for subdir in ["Normal", "Diseased"]:
        if subdir == "Diseased":
            test_image_dir = os.path.join(root_dir, subdir, "Testset_Disease")
        else:
            test_image_dir = os.path.join(root_dir, subdir, f"Testset_{subdir}")
        test_image_paths.extend([os.path.join(test_image_dir, filename) for filename in os.listdir(test_image_dir)])

    test_data = [{"image": image_path} for image_path in test_image_paths]
    test_ds = CacheDataset(data=test_data, transform=test_transforms)
    return test_ds


def extract_number_and_type(filepath):
    # Extract the number and type from the filename
    match = re.search(r'(/Normal/|/Diseased/).*_(\d+).nrrd', filepath)
    if match:
        disease_type = match.group(1)
        number = int(match.group(2))
        return (disease_type, number)
    else:
        return ('', 0)

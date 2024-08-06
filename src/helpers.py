import os
from pandas import pd
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

def create_directories(augmented_dir: str, output_dir: str):
    """
    Create directories if they do not exist.

    This function ensures that the specified directories exist. If they do not exist,
    it creates them.

    :param augmented_dir: Directory to store augmented images.
    :param output_dir: Directory to store output images.
    """
    os.makedirs(augmented_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

def prepare_class_indices(dataframe: pd.DataFrame, class_names: dict) -> dict:
    """
    Prepare indices for each class.

    This function creates a dictionary mapping each class name to a list of indices
    in the dataframe where that class is present.

    :param dataframe: A Pandas DataFrame containing dataset information.
    :param class_names: A dictionary mapping category IDs to class names.
    :return: A dictionary with class names as keys and lists of indices as values.
    """
    return {
        class_name: dataframe[dataframe['name'] == class_name].index.tolist()
        for class_name in class_names.values()
    }

def update_labels_tensor(unique_file_names: list, multi_label_labels_dict: dict, num_classes: int) -> torch.Tensor:
    """
    Update the labels tensor for multi-label classification.

    This function creates a tensor of multi-labels for each file name in the unique file names list.

    :param unique_file_names: A list of unique file names.
    :param multi_label_labels_dict: A dictionary with file names as keys and multi-label lists as values.
    :param num_classes: The number of classes.
    :return: A tensor of multi-labels.
    """
    multi_label_labels_list = [
        torch.tensor(multi_label_labels_dict[file_name])
        for file_name in unique_file_names
        if file_name in multi_label_labels_dict
    ]
    return torch.stack(multi_label_labels_list)

def augment_image(img: np.ndarray, bboxes: list, class_labels: list) -> dict:
    """
    Apply augmentation to the image and bounding boxes.

    This function applies a series of augmentations to the input image and its bounding boxes,
    including horizontal flip and conversion to tensor.

    :param img: The input image as a NumPy array.
    :param bboxes: A list of bounding boxes in COCO format.
    :param class_labels: A list of class labels corresponding to the bounding boxes.
    :return: A dictionary containing the augmented image and bounding boxes.
    """
    augmentation = A.Compose([
        A.HorizontalFlip(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'], clip=True))
    return augmentation(image=img, bboxes=bboxes, class_labels=class_labels)

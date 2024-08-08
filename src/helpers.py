import os

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2


def create_directories(augmented_dir: str):
    """
    Create directories if they do not exist.

    This function ensures that the specified directories exist. If they do not exist,
    it creates them.

    :param augmented_dir: Directory to store augmented images.
    :param output_dir: Directory to store output images.
    """
    os.makedirs(augmented_dir, exist_ok=True)


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
        class_name: dataframe[dataframe["name"] == class_name].index.tolist() for class_name in class_names.values()
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
    augmentation = A.Compose(
        [A.HorizontalFlip(), ToTensorV2()],
        bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"], clip=True),
    )
    return augmentation(image=img, bboxes=bboxes, class_labels=class_labels)


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (list or tuple): Bounding box in the format [x1, y1, x2, y2].
        box2 (list or tuple): Ground truth bounding box in the format [x1, y1, x2, y2].

    Returns:
        float: IoU value.
    """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def calculate_metrics(model, data_loader, device, iou_threshold=0.5):
    """
    Calculate precision and recall metrics for a given model and data loader.

    Args:
        model (torch.nn.Module): Trained object detection model.
        data_loader (torch.utils.data.DataLoader): Data loader for the dataset.
        device (torch.device): Device to run the model on ('cuda' or 'cpu').
        iou_threshold (float): IoU threshold to determine a true positive (default is 0.5).

    Returns:
        tuple: Precision and recall values.
    """
    model.eval()
    tp, fp, fn = 0, 0, 0
    with torch.no_grad():
        for data in data_loader:
            imgs = []
            targets = []

            if len(data) < 2:
                continue

            for img in data[0]:
                imgs.append((torch.tensor(img, dtype=torch.float32)).to(device))

            for target in data[1]:
                boxes = target["boxes"]
                labels = target["labels"]

                # Convert (x1, y1, width, height) to (x1, y1, x2, y2)
                converted_boxes = []
                for box in boxes:
                    x1, y1, w, h = box
                    x2 = x1 + w
                    y2 = y1 + h
                    converted_boxes.append([x1, y1, x2, y2])

                converted_boxes = torch.tensor(converted_boxes, dtype=torch.float32).to(device)
                labels = labels.to(device)

                targ = {
                    "boxes": converted_boxes,
                    "labels": labels,
                }

                targets.append(targ)

            outputs = model(imgs)

            for target, output in zip(targets, outputs):
                gt_boxes = target["boxes"]
                pred_boxes = output["boxes"]
                pred_scores = output["scores"]
                pred_labels = output["labels"]

                for i, gt_box in enumerate(gt_boxes):
                    if len(pred_boxes) == 0:
                        fn += 1
                        continue

                    ious = [calculate_iou(gt_box, pred_box) for pred_box in pred_boxes]
                    max_iou = max(ious)
                    max_iou_idx = ious.index(max_iou)

                    if max_iou >= iou_threshold and pred_labels[max_iou_idx] == target["labels"][i]:
                        tp += 1
                        pred_boxes = torch.cat((pred_boxes[:max_iou_idx], pred_boxes[max_iou_idx + 1 :]))
                    else:
                        fn += 1

                fp += len(pred_boxes)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    return precision, recall


def coco_to_pascal(box):
    """
    Convert a bounding box from COCO format (x1, y1, width, height) to Pascal VOC format (x1, y1, x2, y2).

    Args:
        box (list or tuple): Bounding box in COCO format.

    Returns:
        list: Bounding box in Pascal VOC format.
    """
    x1, y1, w, h = box
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]


def assert_pascal_boxes(boxes):
    """
    Assert that a list of bounding boxes are in Pascal VOC format (x1, y1, x2, y2).

    Args:
        boxes (list): List of bounding boxes.

    Raises:
        ValueError: If any bounding box is not in Pascal VOC format.
    """
    for box in boxes:
        assert_pascal_box(box)


def assert_pascal_box(box):
    """
    Assert that a bounding box is in Pascal VOC format (x1, y1, x2, y2).

    Args:
        box (list or tuple): Bounding box in Pascal VOC format.

    Raises:
        ValueError: If the bounding box is not in Pascal VOC format.
    """
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid box {box}")

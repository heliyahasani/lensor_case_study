import os
from typing import Tuple

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from helpers import (
    assert_pascal_boxes,
    augment_image,
    coco_to_pascal,
    create_directories,
    prepare_class_indices,
    update_labels_tensor,
)


class CustomDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        unique_file_names: list,
        image_directory: str,
        max_class_size: int = 100,
        oversample: bool = True,
        augment: bool = True,
        augmented_dir: str = "/Users/heliyahasani/Desktop/lensor_case_study/dataset/augmented/images/train",
    ):
        """
        Initialize the CustomDataset class.

        :param dataframe: A Pandas DataFrame containing dataset information including 'file_name', 'category_id', 'bbox', etc.
        :param unique_file_names: A list of unique file names present in the dataset.
        :param image_directory: The directory where the original images are stored.
        :param max_class_size: The maximum number of samples per class. Defaults to 100.
        :param oversample: Boolean flag to determine whether to oversample minority classes. Defaults to True.
        :param augment: Boolean flag to determine whether to augment the dataset to balance classes. Defaults to True.
        :param augmented_dir: Directory to store augmented images. Defaults to 'raw/images/train'.
        :param output_dir: Directory to store output images. Defaults to 'output_images'.
        """
        self.dataframe = dataframe
        self.unique_file_names = list(unique_file_names)
        self.image_directory = image_directory
        self.max_class_size = max_class_size
        self.augmented_dir = augmented_dir
        create_directories(self.augmented_dir)

        self.class_names = self.dataframe.set_index("category_id")["name"].to_dict()
        self.class_indices = prepare_class_indices(self.dataframe, self.class_names)
        self.num_classes = len(self.class_names)

        self.multi_label_labels_dict = self._create_multi_label_dict()
        self.multi_label_labels_tensor = update_labels_tensor(
            self.unique_file_names, self.multi_label_labels_dict, self.num_classes
        )

        if augment:
            self.augment_minority_classes()

        self.transform = A.Compose(
            [A.Normalize(normalization="standard")],
            bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"], clip=True),
        )

    def _create_multi_label_dict(self) -> dict:
        """
        Create a dictionary for multi-label classification.

        :return: A dictionary with file names as keys and multi-label lists as values.
        """
        multi_label_labels_dict = {}
        for idx in self.dataframe.index:
            filename = self.dataframe.loc[idx, "file_name"]
            labels = set(self.dataframe[self.dataframe["file_name"] == filename]["name"].tolist())
            multi_label_labels = [0] * self.num_classes
            for label in labels:
                class_id = list(self.class_names.values()).index(label)
                multi_label_labels[class_id] = 1
            multi_label_labels_dict[filename] = multi_label_labels
        return multi_label_labels_dict

    def augment_minority_classes(self):
        """
        Augment minority classes to balance the dataset.
        """
        augmented_count = 0

        for class_name, indices in self.class_indices.items():
            if len(indices) < self.max_class_size:
                num_to_augment = self.max_class_size - len(indices)
                for idx in indices:
                    if num_to_augment <= 0:
                        break
                    file_name = self.dataframe.loc[idx, "file_name"]
                    img_path = os.path.join(self.image_directory, file_name)
                    img = np.array(Image.open(img_path).convert("RGB"))
                    class_labels = [
                        self.class_names[x]
                        for x in self.dataframe[self.dataframe["file_name"] == file_name]["category_id"]
                    ]
                    bboxes = [tuple(x) for x in self.dataframe[self.dataframe["file_name"] == file_name]["bbox"]]
                    for i in range(num_to_augment):
                        augmented = augment_image(img, bboxes, class_labels)
                        augmented_img = augmented["image"]
                        augmented_boxes = np.array(augmented["bboxes"])

                        if len(augmented_boxes) == 0:
                            continue

                        augmented_file_name = f"aug_{class_name}_{i}_{file_name}"
                        augmented_img_path = os.path.join(self.augmented_dir, augmented_file_name)
                        T.ToPILImage()(augmented_img).save(augmented_img_path)

                        self.unique_file_names.append(augmented_file_name)
                        self.multi_label_labels_dict[augmented_file_name] = self.multi_label_labels_dict[file_name]

                        for box in augmented_boxes:
                            new_row = self.dataframe[self.dataframe["file_name"] == file_name].iloc[0].copy()
                            new_row["x1"], new_row["y1"], new_row["w"], new_row["h"] = box
                            new_row["bbox"] = [box[0], box[1], box[2], box[3]]
                            new_row["file_name"] = augmented_file_name
                            self.dataframe = pd.concat([self.dataframe, pd.DataFrame([new_row])], ignore_index=True)

                        num_to_augment -= 1
                        augmented_count += 1

        self.dataframe.reset_index(drop=True, inplace=True)
        self.unique_file_names = self.dataframe["file_name"].unique().tolist()

        self.dataframe.to_csv("augmented_data.csv", index=False)

        self.multi_label_labels_tensor = update_labels_tensor(
            self.unique_file_names, self.multi_label_labels_dict, self.num_classes
        )

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        :return: The number of samples.
        """
        return len(self.unique_file_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Get a sample from the dataset.

        :param idx: Index of the sample.
        :return: A tuple containing the transformed image and target dictionary with bounding boxes, labels, image_id, area, and iscrowd.
        """
        file_name = self.unique_file_names[idx]
        img_path = (
            os.path.join(self.image_directory, file_name)
            if os.path.exists(os.path.join(self.image_directory, file_name))
            else os.path.join(self.augmented_dir, file_name)
        )
        # Extract relevant rows for the current image
        df_subset = self.dataframe[self.dataframe["file_name"] == file_name]

        # Extract bboxes, labels, area, and iscrowd
        bboxes = df_subset["bbox"].tolist()
        labels = df_subset["category_id"].tolist()  # Use category_id directly, assuming it is already integer
        areas = df_subset["area"].tolist()
        iscrowd = df_subset["iscrowd"].tolist()

        # Ensure consistency in the number of bounding boxes and labels
        assert (
            len(bboxes) == len(labels) == len(areas) == len(iscrowd)
        ), "Mismatch in the number of bboxes, labels, areas, and iscrowd values"

        img = np.array(Image.open(img_path).convert("RGB"))
        transformed = self.transform(
            image=img,
            bboxes=bboxes,
            class_labels=labels,
        )
        transformed_img = transformed["image"]
        transformed_img = torch.tensor(transformed_img, dtype=torch.float32).permute(2, 0, 1)
        pascal_boxes = [list(coco_to_pascal(box)) for box in transformed["bboxes"]]
        assert_pascal_boxes(pascal_boxes)

        # Extract image_id from the dataframe
        image_id = df_subset["image_id"].iloc[0]

        target = {
            "boxes": torch.tensor(np.array(pascal_boxes), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),  # Labels should be integer indices
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }

        return transformed_img, target

    def get_labels(self, idx: int) -> torch.Tensor:
        """
        Get the multi-label tensor for a given sample.

        :param idx: Index of the sample.
        :return: Multi-label tensor.
        """
        return self.multi_label_labels_tensor[idx]


def create_dataset_(
    dataframe: pd.DataFrame,
    unique_file_names: list,
    image_directory: str,
    max_class_size: int = 100,
    augment: bool = True,
) -> CustomDataset:
    """
    Prefect task to create a CustomDataset instance.

    This function creates an instance of the CustomDataset class, which is used to handle
    and preprocess image data for machine learning tasks. The dataset includes functionality
    for data augmentation to balance class distributions.

    :param dataframe: A Pandas DataFrame containing the dataset information.
                      It should include columns such as 'file_name', 'category_id', 'bbox', etc.
    :param unique_file_names: A list of unique file names present in the dataset.
    :param image_directory: The directory where the original images are stored.
    :param max_class_size: The maximum number of samples per class. Defaults to 100.
    :param augment: Boolean flag to determine whether to augment the dataset to balance classes. Defaults to True.
    :return: An instance of the CustomDataset class.
    """
    return CustomDataset(dataframe, unique_file_names, image_directory, max_class_size, augment=augment)

import json
import logging
from typing import Tuple

import pandas as pd


class DataPreparation:
    def __init__(self):
        """
        Initialize the DataPreparation class with JSON file paths.

        :param train_json: Path to the training data JSON file.
        :param val_json: Path to the validation data JSON file.
        :param test_json: Path to the test data JSON file.
        """
        self.train_json = "/Users/heliyahasani/Desktop/lensor_case_study/dataset/annotations/instances_train.json"
        self.val_json = "/Users/heliyahasani/Desktop/lensor_case_study/dataset/annotations/instances_val.json"
        self.test_json = "/Users/heliyahasani/Desktop/lensor_case_study/dataset/annotations/instances_test.json"
        self.logger = logging.getLogger(__name__)

    def load_json_to_dataframe(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load JSON data and convert it to Pandas DataFrames.

        :param file_path: Path to the JSON file.
        :return: A tuple containing DataFrames for categories, images, and annotations.
        """
        self.logger.info(f"Loading JSON data from {file_path}")
        with open(file_path, "r") as file:
            data = json.load(file)
        categories_df = pd.DataFrame(data["categories"])
        images_df = pd.DataFrame(data["images"])
        annotations_df = pd.DataFrame(data["annotations"])
        return categories_df, images_df, annotations_df

    def merge_dataframes(
        self, images_df: pd.DataFrame, annotations_df: pd.DataFrame, categories_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge the images, annotations, and categories DataFrames.

        :param images_df: DataFrame containing image data.
        :param annotations_df: DataFrame containing annotation data.
        :param categories_df: DataFrame containing category data.
        :return: A merged DataFrame.
        """
        self.logger.info("Merging dataframes")
        merged_df = pd.merge(
            annotations_df, images_df, left_on="image_id", right_on="id", suffixes=("_annotation", "_image")
        )
        merged_df = pd.merge(
            merged_df, categories_df, left_on="category_id", right_on="id", suffixes=("_merged", "_category")
        )
        return merged_df

    def add_bbox_coordinates(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add bounding box coordinates to the merged DataFrame.

        :param merged_df: The merged DataFrame.
        :return: The DataFrame with bounding box coordinates added.
        """
        self.logger.info("Adding bounding box coordinates")
        box_coordinates = pd.DataFrame(merged_df["bbox"].tolist(), columns=["x1", "y1", "w", "h"], dtype=float)
        merged_df = pd.concat([merged_df, box_coordinates], axis=1)
        return merged_df

    def get_balanced_samples(self, dataframe: pd.DataFrame, max_samples_per_category: int = 100) -> pd.DataFrame:
        """
        Balance the samples by category.

        :param dataframe: The DataFrame to balance.
        :param max_samples_per_category: Maximum samples per category.
        :return: A balanced DataFrame.
        """
        self.logger.info(f"Balancing samples with max {max_samples_per_category} per category")
        balanced_df = (
            dataframe.groupby("category_id")
            .apply(lambda x: x.sample(min(len(x), max_samples_per_category)))
            .reset_index(drop=True)
        )
        return balanced_df

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare the training, validation, and test data.

        :return: A tuple containing DataFrames for training, validation, test, and balanced training data.
        """
        self.logger.info("Preparing training data")
        train_categories_df, train_images_df, train_annotations_df = self.load_json_to_dataframe(self.train_json)
        self.logger.info("Preparing validation data")
        validation_categories_df, validation_images_df, validation_annotations_df = self.load_json_to_dataframe(
            self.val_json
        )
        self.logger.info("Preparing test data")
        test_categories_df, test_images_df, test_annotations_df = self.load_json_to_dataframe(self.test_json)

        self.logger.info("Merging training data")
        train_merged_df = self.merge_dataframes(train_images_df, train_annotations_df, train_categories_df)
        self.logger.info("Merging validation data")
        validation_merged_df = self.merge_dataframes(
            validation_images_df, validation_annotations_df, validation_categories_df
        )
        self.logger.info("Merging test data")
        test_merged_df = self.merge_dataframes(test_images_df, test_annotations_df, test_categories_df)

        self.logger.info("Adding bounding box coordinates to training data")
        train_merged_df = self.add_bbox_coordinates(train_merged_df)
        self.logger.info("Adding bounding box coordinates to validation data")
        validation_merged_df = self.add_bbox_coordinates(validation_merged_df)
        self.logger.info("Adding bounding box coordinates to test data")
        test_merged_df = self.add_bbox_coordinates(test_merged_df)

        self.logger.info("Balancing training data")
        balanced_df = self.get_balanced_samples(train_merged_df, max_samples_per_category=400)

        return train_merged_df, validation_merged_df, test_merged_df, balanced_df

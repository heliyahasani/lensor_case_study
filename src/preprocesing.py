import json
import pandas as pd

class ImageAnnotationPreProcessor:
    """
    A class to process image annotations from JSON files.
    """

    def __init__(self, train_path=None, val_path=None, test_path=None):
        """
        Initialize the ImageAnnotationPreProcessor with file paths.

        :param train_path: Path to the training data JSON file.
        :param val_path: Path to the validation data JSON file.
        :param test_path: Path to the test data JSON file.
        """
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.merged_dataframes = {}  # Dictionary to store merged DataFrames

    def load_json_to_dataframe(self, file_path):
        """
        Load JSON data from a file and convert to pandas DataFrames.

        :param file_path: Path to the JSON file.
        :return: Tuple of pandas DataFrames (categories_df, images_df, annotations_df)
        """
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        categories_df = pd.DataFrame(data['categories'])
        images_df = pd.DataFrame(data['images'])
        annotations_df = pd.DataFrame(data['annotations'])
        
        return categories_df, images_df, annotations_df

    def merge_dataframes(self, images_df, annotations_df, categories_df):
        """
        Merge the images, annotations, and categories DataFrames.

        :param images_df: DataFrame containing images data.
        :param annotations_df: DataFrame containing annotations data.
        :param categories_df: DataFrame containing categories data.
        :return: Merged DataFrame.
        """
        merged_df = pd.merge(annotations_df, images_df, left_on='image_id', right_on='id', suffixes=('_annotation', '_image'))
        merged_df = pd.merge(merged_df, categories_df, left_on='category_id', right_on='id', suffixes=('_merged', '_category'))
        return merged_df

    def add_bbox_coordinates(self, merged_df):
        """
        Extract bounding box coordinates from the 'bbox' column and add them as separate columns.

        :param merged_df: Merged DataFrame containing the 'bbox' column.
        :return: DataFrame with separate columns for bounding box coordinates.
        """
        box_coordinates = pd.DataFrame(merged_df['bbox'].tolist(), columns=['x1', 'y1', 'w', 'h'], dtype=float)
        merged_df = pd.concat([merged_df, box_coordinates], axis=1)
        merged_df["x2"] = merged_df["x1"] + merged_df["w"]
        merged_df["y2"] = merged_df["y1"] + merged_df["h"]
        return merged_df

    def process_and_get_dataframe(self, data_type='train'):
        """
        Process the data based on the specified type (train, val, or test) and return the merged DataFrame.

        :param data_type: Type of data to process ('train', 'val', or 'test').
        :return: Merged DataFrame.
        """
        if data_type == 'train' and self.train_path:
            categories_df, images_df, annotations_df = self.load_json_to_dataframe(self.train_path)
            self.merged_dataframes['train'] = self.merge_dataframes(images_df, annotations_df, categories_df)
            self.merged_dataframes['train'] = self.add_bbox_coordinates(self.merged_dataframes['train'])
        elif data_type == 'val' and self.val_path:
            categories_df, images_df, annotations_df = self.load_json_to_dataframe(self.val_path)
            self.merged_dataframes['val'] = self.merge_dataframes(images_df, annotations_df, categories_df)
            self.merged_dataframes['val'] = self.add_bbox_coordinates(self.merged_dataframes['val'])
        elif data_type == 'test' and self.test_path:
            categories_df, images_df, annotations_df = self.load_json_to_dataframe(self.test_path)
            self.merged_dataframes['test'] = self.merge_dataframes(images_df, annotations_df, categories_df)
            self.merged_dataframes['test'] = self.add_bbox_coordinates(self.merged_dataframes['test'])
        else:
            raise ValueError("Invalid data type or file path not provided")

        return self.merged_dataframes[data_type]

    def get_unique_image_filenames(self, data_type='train'):
        """
        Get unique image filenames from the specified data type (train, val, or test).

        :param data_type: Type of data to get unique image filenames from ('train', 'val', or 'test').
        :return: Array of unique image filenames.
        """
        if data_type in self.merged_dataframes:
            return self.merged_dataframes[data_type]['file_name'].unique()
        else:
            raise ValueError("Data not processed or invalid data type")
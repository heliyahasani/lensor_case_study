import json
import unittest
import pandas as pd
from unittest.mock import patch, mock_open

from preprocessing import ImageAnnotationPreProcessor

class TestImageAnnotationPreProcessor(unittest.TestCase):

    def setUp(self):
        self.sample_json = {
            "categories": [
                {"id": 1, "name": "category1"},
                {"id": 2, "name": "category2"}
            ],
            "images": [
                {"id": 1, "file_name": "image1.jpg"},
                {"id": 2, "file_name": "image2.jpg"}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]},
                {"id": 2, "image_id": 2, "category_id": 2, "bbox": [15, 25, 35, 45]}
            ]
        }

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({
        "categories": [
            {"id": 1, "name": "category1"},
            {"id": 2, "name": "category2"}
        ],
        "images": [
            {"id": 1, "file_name": "image1.jpg"},
            {"id": 2, "file_name": "image2.jpg"}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [15, 25, 35, 45]}
        ]
    }))
    def test_load_json_to_dataframe(self, mock_file):
        processor = ImageAnnotationPreProcessor()
        categories_df, images_df, annotations_df = processor.load_json_to_dataframe("dummy_path.json")
        self.assertEqual(len(categories_df), 2)
        self.assertEqual(len(images_df), 2)
        self.assertEqual(len(annotations_df), 2)

    def test_merge_dataframes(self):
        processor = ImageAnnotationPreProcessor()
        categories_df = pd.DataFrame(self.sample_json["categories"])
        images_df = pd.DataFrame(self.sample_json["images"])
        annotations_df = pd.DataFrame(self.sample_json["annotations"])
        merged_df = processor.merge_dataframes(images_df, annotations_df, categories_df)
        self.assertEqual(len(merged_df), 2)
        self.assertIn("file_name", merged_df.columns)
        self.assertIn("bbox", merged_df.columns)

    def test_add_bbox_coordinates(self):
        processor = ImageAnnotationPreProcessor()
        merged_df = pd.DataFrame([{
            "bbox": [10, 20, 30, 40],
            "other_column": "value"
        }])
        result_df = processor.add_bbox_coordinates(merged_df)
        self.assertIn("x1", result_df.columns)
        self.assertIn("y1", result_df.columns)
        self.assertIn("x2", result_df.columns)
        self.assertIn("y2", result_df.columns)
        self.assertEqual(result_df.iloc[0]["x2"], 40)
        self.assertEqual(result_df.iloc[0]["y2"], 60)

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({
        "categories": [
            {"id": 1, "name": "category1"},
            {"id": 2, "name": "category2"}
        ],
        "images": [
            {"id": 1, "file_name": "image1.jpg"},
            {"id": 2, "file_name": "image2.jpg"}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [15, 25, 35, 45]}
        ]
    }))
    def test_process_and_get_dataframe(self, mock_file):
        processor = ImageAnnotationPreProcessor(train_path="dummy_path.json")
        train_df = processor.process_and_get_dataframe(data_type='train')
        self.assertEqual(len(train_df), 2)
        self.assertIn("file_name", train_df.columns)
        self.assertIn("x1", train_df.columns)

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({
        "categories": [
            {"id": 1, "name": "category1"},
            {"id": 2, "name": "category2"}
        ],
        "images": [
            {"id": 1, "file_name": "image1.jpg"},
            {"id": 2, "file_name": "image2.jpg"}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [15, 25, 35, 45]}
        ]
    }))
    def test_get_unique_image_filenames(self, mock_file):
        processor = ImageAnnotationPreProcessor(train_path="dummy_path.json")
        processor.process_and_get_dataframe(data_type='train')
        unique_images = processor.get_unique_image_filenames(data_type='train')
        self.assertEqual(len(unique_images), 2)
        self.assertIn("image1.jpg", unique_images)
        self.assertIn("image2.jpg", unique_images)

if __name__ == "__main__":
    unittest.main()

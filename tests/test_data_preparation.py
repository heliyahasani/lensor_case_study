import json
import logging
import os

import pytest
from prefect import Flow

from prepare import (  
    DataPreparation,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory for test files
test_dir = 'testing'

# Create the testing directory if it does not exist
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

img1  = "dataset/train/_1OL6RBGTRaIlXPqpsNXCA_jpeg.rf.2e0aff71cdeb0fc5615a484010eb3c56.jpg"
img2  = "dataset/train/_1OL6RBGTRaIlXPqpsNXCA_jpeg.rf.6cfdecf9f081672f8a9f1c91c75d1ab9.jpg"

# Sample JSON data for testing
sample_data = {
    "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
    "images": [{"id": 1, "file_name": img1}, {"id": 2, "file_name": img2}],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]},
        {"id": 2, "image_id": 2, "category_id": 2, "bbox": [50, 60, 70, 80]}
    ]
}

# Save sample data to json files inside the testing directory
train_json = os.path.join(test_dir, 'train.json')
val_json = os.path.join(test_dir, 'val.json')
test_json = os.path.join(test_dir, 'test.json')

with open(train_json, 'w') as file:
    json.dump(sample_data, file)

with open(val_json, 'w') as file:
    json.dump(sample_data, file)

with open(test_json, 'w') as file:
    json.dump(sample_data, file)

@pytest.fixture
def data_preparation():
    return DataPreparation(train_json, val_json, test_json)

def test_load_json_to_dataframe(data_preparation):
    categories_df, images_df, annotations_df = data_preparation.load_json_to_dataframe(train_json)
    
    assert not categories_df.empty
    assert not images_df.empty
    assert not annotations_df.empty
    assert list(categories_df.columns) == ["id", "name"]
    assert list(images_df.columns) == ["id", "file_name"]
    assert list(annotations_df.columns) == ["id", "image_id", "category_id", "bbox"]

def test_merge_dataframes(data_preparation):
    categories_df, images_df, annotations_df = data_preparation.load_json_to_dataframe(train_json)
    merged_df = data_preparation.merge_dataframes(images_df, annotations_df, categories_df)
    
    assert "file_name" in merged_df.columns
    assert "name" in merged_df.columns
    assert "bbox" in merged_df.columns

def test_add_bbox_coordinates(data_preparation):
    categories_df, images_df, annotations_df = data_preparation.load_json_to_dataframe(train_json)
    merged_df = data_preparation.merge_dataframes(images_df, annotations_df, categories_df)
    merged_df_with_bbox = data_preparation.add_bbox_coordinates(merged_df)
    
    assert "x1" in merged_df_with_bbox.columns
    assert "y1" in merged_df_with_bbox.columns
    assert "w" in merged_df_with_bbox.columns
    assert "h" in merged_df_with_bbox.columns

def test_get_balanced_samples(data_preparation):
    categories_df, images_df, annotations_df = data_preparation.load_json_to_dataframe(train_json)
    merged_df = data_preparation.merge_dataframes(images_df, annotations_df, categories_df)
    merged_df_with_bbox = data_preparation.add_bbox_coordinates(merged_df)
    balanced_df = data_preparation.get_balanced_samples(merged_df_with_bbox, max_samples_per_category=1)
    
    assert len(balanced_df) == 2  # Since there are 2 categories and max_samples_per_category is 1

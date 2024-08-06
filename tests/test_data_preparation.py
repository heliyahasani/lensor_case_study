import os
import pytest
import json
import logging
from prefect import Flow
from prepare import DataPreparation, prepare_data_task  # Replace with the actual module name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory for test files
test_dir = 'testing'

# Create the testing directory if it does not exist
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Sample JSON data for testing
sample_data = {
    "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
    "images": [{"id": 1, "file_name": "image1.jpg"}, {"id": 2, "file_name": "image2.jpg"}],
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

def test_prepare_data_task():
    with Flow("test_flow") as flow:
        train_df, val_df, test_df, balanced_df = prepare_data_task(train_json, val_json, test_json)
    
    state = flow.run()
    assert state.is_successful()
    task_state = state.result[prepare_data_task]
    
    assert isinstance(task_state.result, tuple)
    assert len(task_state.result) == 4

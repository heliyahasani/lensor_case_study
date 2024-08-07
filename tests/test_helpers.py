import os

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from helpers import (
    augment_image,
    create_directories,
    prepare_class_indices,
    update_labels_tensor,
)


@pytest.fixture
def sample_dataframe():
    data = {
        'file_name': ['image1.jpg', 'image2.jpg'],
        'category_id': [1, 2],
        'name': ['cat', 'dog'],
        'bbox': [[10, 20, 30, 40], [50, 60, 70, 80]]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_multi_label_dict():
    return {
        'image1.jpg': [1, 0],
        'image2.jpg': [0, 1]
    }

@pytest.fixture
def sample_image():
    return np.array(Image.new('RGB', (100, 100), color=(73, 109, 137)))

def test_create_directories(tmpdir):
    augmented_dir = os.path.join(tmpdir, 'augmented')
    output_dir = os.path.join(tmpdir, 'output')

    create_directories(augmented_dir, output_dir)

    assert os.path.exists(augmented_dir)
    assert os.path.exists(output_dir)

def test_prepare_class_indices(sample_dataframe):
    class_names = {1: 'cat', 2: 'dog'}
    class_indices = prepare_class_indices(sample_dataframe, class_names)
    
    assert class_indices == {'cat': [0], 'dog': [1]}

def test_update_labels_tensor(sample_multi_label_dict):
    unique_file_names = ['image1.jpg', 'image2.jpg']
    num_classes = 2
    labels_tensor = update_labels_tensor(unique_file_names, sample_multi_label_dict, num_classes)
    
    expected_tensor = torch.tensor([[1, 0], [0, 1]])
    assert torch.equal(labels_tensor, expected_tensor)

def test_augment_image(sample_image):
    bboxes = [[10, 20, 30, 40]]
    class_labels = ['cat']
    augmented_data = augment_image(sample_image, bboxes, class_labels)
    
    assert 'image' in augmented_data
    assert 'bboxes' in augmented_data
    assert isinstance(augmented_data['image'], torch.Tensor)
    assert isinstance(augmented_data['bboxes'], list)
    assert len(augmented_data['bboxes']) > 0

if __name__ == '__main__':
    pytest.main([__file__])

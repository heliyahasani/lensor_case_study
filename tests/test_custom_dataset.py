import unittest
from io import BytesIO

import numpy as np
import pandas as pd
import torch
from PIL import Image

from custom_dataset import CustomDataset

img1  = "/train/_1OL6RBGTRaIlXPqpsNXCA_jpeg.rf.2e0aff71cdeb0fc5615a484010eb3c56.jpg"
img2  = "/train/_1OL6RBGTRaIlXPqpsNXCA_jpeg.rf.6cfdecf9f081672f8a9f1c91c75d1ab9.jpg"
img3  = "/train/_1OL6RBGTRaIlXPqpsNXCA_jpeg.rf.fcf788371d618e45d85f008c64238770.jpg"
img4  = "/train/1_jpg.rf.5a99d7429da2a3ab9bd88105a31d9c4a.jpg"

class TestCustomDataset(unittest.TestCase):

    def setUp(self):
        # Sample dataframe setup
        data = {
            'category_id': [0, 1, 0, 1],
            'name': ['d1', 'd2', 'd1', 'd2'],
            'file_name': [img1,img2,img3,img4],
            'x1': [10, 15, 10, 15],
            'y1': [20, 25, 20, 25],
            'x2': [110, 115, 110, 115],
            'y2': [120, 125, 120, 125]
        }
        self.dataframe = pd.DataFrame(data)
        self.unique_file_names = [img1,img2,img3,img4]
        self.image_directory = 'dataset/images'  # Assuming images are stored in this directory

        # Mocking image files
        self.image_data = {}
        for file_name in self.unique_file_names:
            img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='JPEG')
            self.image_data[file_name] = img_byte_arr

    def test_initialization(self):
        dataset = CustomDataset(self.dataframe, self.unique_file_names, self.image_directory)
        self.assertEqual(len(dataset), len(self.unique_file_names))
        self.assertEqual(dataset.num_classes, 2)
        self.assertIn('d1', dataset.class_names.values())
        self.assertIn('d2', dataset.class_names.values())

    def test_len(self):
        dataset = CustomDataset(self.dataframe, self.unique_file_names, self.image_directory)
        self.assertEqual(len(dataset), len(self.unique_file_names))

    def test_getitem(self):
        dataset = CustomDataset(self.dataframe, self.unique_file_names, self.image_directory)
        
        for idx in range(len(dataset)):
            img, target = dataset[idx]
            self.assertTrue(torch.is_tensor(img))
            self.assertTrue(torch.is_tensor(target['boxes']))
            self.assertTrue(torch.is_tensor(target['labels']))
            self.assertEqual(target['boxes'].shape[1], 4)  # Check if boxes have 4 coordinates

    def test_get_labels(self):
        dataset = CustomDataset(self.dataframe, self.unique_file_names, self.image_directory)
        
        for idx in range(len(dataset)):
            labels = dataset.get_labels(idx)
            self.assertTrue(torch.is_tensor(labels))
            self.assertEqual(labels.shape[0], dataset.num_classes)

    def tearDown(self):
        # Clean up any mock data if needed
        pass

if __name__ == '__main__':
    unittest.main()

import random
import torch
from PIL import Image
from torchvision import transforms as T

class CustomDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for handling multi-label image data.

    Attributes:
        dataframe (pd.DataFrame): DataFrame containing image metadata including file names and labels.
        unique_file_names (list): List of unique image file names.
        image_directory (str): Path to the directory containing image files.
        class_names (dict): Dictionary mapping category IDs to class names.
        class_indices (dict): Dictionary mapping class names to list of indices in the DataFrame.
        num_classes (int): Total number of unique classes.
        multi_label_labels_dict (dict): Dictionary mapping file names to their multi-label encoded vectors.
        multi_label_labels_list (list): List of tensors corresponding to multi-label encoded vectors.
        multi_label_labels_tensor (torch.Tensor): Stacked tensor of multi-label encoded vectors.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Returns the image and target dictionary for the given index.
        get_labels(idx): Returns the multi-label tensor for the given index.
    """
    
    def __init__(self, dataframe, unique_file_names, image_directory):
        """
        Initializes the CustomDataset with the given DataFrame, unique file names, and image directory.

        Args:
            dataframe (pd.DataFrame): DataFrame containing image metadata including file names and labels.
            unique_file_names (list): List of unique image file names.
            image_directory (str): Path to the directory containing image files.
        """
        self.dataframe = dataframe
        self.unique_file_names = unique_file_names
        self.image_directory = image_directory

        # Define class names and create indices for each class
        self.class_names = self.dataframe.set_index('category_id')['name'].to_dict()
        self.class_indices = {
            class_name: self.dataframe[self.dataframe['name'] == class_name].index.tolist() 
            for class_name in self.class_names.values()
        }
        self.num_classes = len(self.class_names)

        # Randomly select images for each class
        selected_indices = []
        for class_name, indices in self.class_indices.items():
            selected_indices.extend(random.sample(indices, min(len(indices), len(self.unique_file_names))))

        # Ensure selected indices are within bounds
        selected_indices = [idx for idx in selected_indices if idx < len(self.dataframe)]

        # Create multi-label encoded dictionary
        self.multi_label_labels_dict = {}
        for idx in selected_indices:
            filename = self.dataframe.loc[idx, 'file_name']
            labels = set(self.dataframe[self.dataframe['file_name'] == filename]['name'].tolist())
            multi_label_labels = [0] * self.num_classes
            for label in labels:
                class_id = list(self.class_names.values()).index(label)
                multi_label_labels[class_id] = 1
            self.multi_label_labels_dict[filename] = multi_label_labels

        # Create a list of tensors corresponding to the unique filenames
        self.multi_label_labels_list = [
            torch.tensor(self.multi_label_labels_dict[file_name]) 
            for file_name in self.unique_file_names 
            if file_name in self.multi_label_labels_dict
        ]
        self.multi_label_labels_tensor = torch.stack(self.multi_label_labels_list)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.multi_label_labels_tensor)

    def __getitem__(self, idx):
        """
        Returns the image and target dictionary for the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, target) where image is a tensor and target is a dictionary containing 'boxes' and 'labels'.
        """
        file_name = self.unique_file_names[idx]
        file_df = self.dataframe[self.dataframe.file_name == file_name]
        img = Image.open(f"{self.image_directory}/{file_name}").convert('RGB')
        boxes = file_df[["x1", "y1", "x2", "y2"]].values.astype("float")
        label = self.multi_label_labels_tensor[idx]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": label  # label is already a tensor
        }
        return T.ToTensor()(img), target

    def get_labels(self, idx):
        """
        Returns the multi-label tensor for the given index.

        Args:
            idx (int): Index of the label tensor to retrieve.

        Returns:
            torch.Tensor: Multi-label tensor for the given index.
        """
        return self.multi_label_labels_tensor[idx]

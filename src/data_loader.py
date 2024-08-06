import torch
from torch.utils.data import DataLoader
from typing import Optional
from custom_dataset import CustomDataset
import utils

class DataLoaderManager:
    """
    A class to manage the creation of PyTorch DataLoader objects.

    Attributes:
    ----------
    custom_dataset : CustomDataset
        The custom dataset to be loaded.
    batch_size : int
        The number of samples per batch to load.
    collate_fn : callable
        Merges a list of samples to form a mini-batch of Tensor(s).
    pin_memory : bool
        If `True`, the data loader will copy Tensors into CUDA pinned memory before returning them.

    Methods:
    -------
    create_data_loader(shuffle=True) -> DataLoader
        Creates and returns a DataLoader object for the dataset.
    """

    def __init__(self, custom_dataset: CustomDataset, batch_size: int, pin_memory: Optional[bool] = None):
        """
        Constructs all the necessary attributes for the DataLoaderManager object.

        Parameters:
        ----------
        custom_dataset : CustomDataset
            The custom dataset to be loaded.
        batch_size : int
            The number of samples per batch to load.
        pin_memory : bool, optional
            If `True`, the data loader will copy Tensors into CUDA pinned memory before returning them.
            If `None`, it defaults to `True` if CUDA is available, otherwise `False`.
        """
        self.custom_dataset = custom_dataset
        self.batch_size = batch_size
        self.collate_fn = utils.collate_fn
        self.pin_memory = pin_memory if pin_memory is not None else torch.cuda.is_available()

    def create_data_loader(self, shuffle: bool = True) -> DataLoader:
        """
        Creates and returns a DataLoader object for the dataset.

        Parameters:
        ----------
        shuffle : bool, optional
            If `True`, data will be reshuffled at every epoch. Default is `True`.

        Returns:
        -------
        DataLoader
            A PyTorch DataLoader object for the custom dataset.
        """
        return DataLoader(
            self.custom_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory
        )

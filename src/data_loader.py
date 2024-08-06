import torch
from torch.utils.data import DataLoader
from prefect import task
import utils


class DataLoaderManager:
    def __init__(self, custom_dataset, batch_size, pin_memory=None):
        self.custom_dataset = custom_dataset
        self.batch_size = batch_size
        self.collate_fn = utils.collate_fn
        self.pin_memory = pin_memory if pin_memory is not None else torch.cuda.is_available()

    def create_data_loader(self, shuffle=True):
        return DataLoader(
            self.custom_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory
        )
def create_data_loader_task_(custom_dataset, batch_size,shuffle=True):
    manager = DataLoaderManager(custom_dataset, batch_size,shuffle)
    return manager.create_data_loader(shuffle=shuffle)

@task
def create_data_loader_task(custom_dataset, batch_size,shuffle=True):
    manager = DataLoaderManager(custom_dataset, batch_size,shuffle)
    return manager.create_data_loader(shuffle=shuffle)

import json
import pandas as pd
from prepare import DataPreparation
from prefect import task, flow
from custom_dataset import create_dataset_
from data_loader import create_data_loader_task_
from model import train_model_task_


train_image_directory = "/Users/heliyahasani/Desktop/lensor_case_study/dataset/images/train"
val_image_directory = "/Users/heliyahasani/Desktop/lensor_case_study/dataset/images/val"
test_image_directory = "/Users/heliyahasani/Desktop/lensor_case_study/dataset/images/test"

train_annotations_json_path = "/Users/heliyahasani/Desktop/lensor_case_study/dataset/annotations/instances_train.json"
val_annotations_json_path = "/Users/heliyahasani/Desktop/lensor_case_study/dataset/annotations/instances_val.json"
test_annotations_json_path = "/Users/heliyahasani/Desktop/lensor_case_study/dataset/annotations/instances_test.json"

data_prep = DataPreparation(train_annotations_json_path,val_annotations_json_path,test_annotations_json_path)
train_merged_df, validation_merged_df, test_merged_df, balanced_df = data_prep.prepare_data()

train_dataset = create_dataset_(train_merged_df, train_merged_df['file_name'].unique(), train_image_directory)
val_dataset = create_dataset_(validation_merged_df, validation_merged_df['file_name'].unique(), val_image_directory, augment=False)
test_dataset = create_dataset_(test_merged_df, test_merged_df['file_name'].unique(), test_image_directory, augment=False)

train_data_loader = create_data_loader_task_(train_dataset, batch_size=4, shuffle=True)
val_data_loader = create_data_loader_task_(val_dataset, batch_size=4, shuffle=False)
test_data_loader = create_data_loader_task_(test_dataset, batch_size=4, shuffle=False)

print("Running model..")
model = train_model_task_(train_data_loader, val_data_loader, num_epochs=1)

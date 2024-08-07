import torch
import torchvision
from torch.utils.data import DataLoader
import utils
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.tensorboard import SummaryWriter

from engine import train_one_epoch, evaluate
from prepare import DataPreparation
from custom_dataset import create_dataset_

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

train_image_directory = "/Users/heliyahasani/Desktop/lensor_case_study/dataset/images/train"
val_image_directory = "/Users/heliyahasani/Desktop/lensor_case_study/dataset/images/val"
test_image_directory = "/Users/heliyahasani/Desktop/lensor_case_study/dataset/images/test"

train_annotations_json_path = "/Users/heliyahasani/Desktop/lensor_case_study/dataset/annotations/instances_train.json"
val_annotations_json_path = "/Users/heliyahasani/Desktop/lensor_case_study/dataset/annotations/instances_val.json"
test_annotations_json_path = "/Users/heliyahasani/Desktop/lensor_case_study/dataset/annotations/instances_test.json"

def main():
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="runs/fasterrcnn_experiment")

    data_prep = DataPreparation(train_annotations_json_path, val_annotations_json_path, test_annotations_json_path)
    train_merged_df, validation_merged_df, test_merged_df, balanced_df = data_prep.prepare_data()

    train_dataset = create_dataset_(train_merged_df, train_merged_df['file_name'].unique(), train_image_directory)
    val_dataset = create_dataset_(validation_merged_df, validation_merged_df['file_name'].unique(), val_image_directory, augment=False)
    test_dataset = create_dataset_(test_merged_df, test_merged_df['file_name'].unique(), test_image_directory, augment=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    data_loader_val = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    # Initialize the model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 9
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    model.to(device)

    # Initialize optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Initialize learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20, scaler=None)
        writer.add_scalar('Loss/train', train_loss.meters['loss'].global_avg, epoch)
        lr_scheduler.step()
        coco_evaluator = evaluate(model, data_loader_val, device)  # Evaluate on validation set
        for i, stat in enumerate(coco_evaluator.coco_eval['bbox'].stats):
            writer.add_scalar(f'COCOEval/Val_{i}', stat, epoch)

    coco_evaluator = evaluate(model, data_loader_test, device)  # Final evaluation on test set
    for i, stat in enumerate(coco_evaluator.coco_eval['bbox'].stats):
        writer.add_scalar(f'COCOEval/Test_{i}', stat, num_epochs)

    torch.save(model.state_dict(), "fasterrcnn_model.pth")

    writer.close()

if __name__ == '__main__':
    main()
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from prefect import task
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset


class ModelTrainer:
    def __init__(self, num_classes=9, learning_rate=0.001, momentum=0.9, weight_decay=0.0005, step_size=3, gamma=0.1):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        
        # Initialize the model, optimizer, and learning rate scheduler
        
        self.device = self.get_device()
        self.model = self._build_model()
        self.model.to(self.device)
        self.optimizer = self._build_optimizer(self.model, learning_rate, momentum, weight_decay)
        self.lr_scheduler = self._build_lr_scheduler(self.optimizer, step_size, gamma)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter()

    def _build_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model

    def get_device(self):
        return torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    def _build_optimizer(self, model, learning_rate, momentum, weight_decay):
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        return optimizer

    def _build_lr_scheduler(self, optimizer, step_size, gamma):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        return lr_scheduler

    def print_model_summary(self):
        print(self.model)

    def train(self, train_data_loader, val_data_loader, num_epochs):
        for epoch in range(num_epochs):
            print(f"Running epoch: {epoch}")
            self.model.train()
            epoch_loss = 0
            batch_no = 0

            progress_bar = tqdm(train_data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
            for i, data in enumerate(progress_bar):
                imgs, targets = self._build_model_input(data)
                if imgs is None:
                    continue
                
                if any(len(t['boxes']) > 0 for t in targets):
                    loss_dict = self.model(imgs, targets)
                    loss = sum(v for v in loss_dict.values())
                    epoch_loss += loss.item()
                else:
                    print(" No positive proposals in this batch.")
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_no +=1
                
                progress_bar.set_postfix(loss=loss.item())


            self.lr_scheduler.step()
            epoch_loss /= batch_no
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss}")

            # Log training loss to TensorBoard
            self.writer.add_scalar('Loss/train', epoch_loss, epoch )

            val_loss = self.evaluate(val_data_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}")

            self.writer.add_scalar('Loss/val', val_loss, epoch)

        self.writer.close()

    def evaluate(self, data_loader):
        self.model.eval()
        coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = ['bbox']
        coco_evaluator = CocoEvaluator(coco, iou_types)
        with torch.no_grad():
            for data in data_loader:
                imgs, targets = self._build_model_input(data)
                if imgs is None:
                    continue

                outputs = self.model(imgs)

                outputs = [{k: v.to(self.device) for k, v in t.items()} for t in outputs]

                res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
                coco_evaluator.update(res)
        
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        return coco_evaluator
    
    def _build_model_input(self, data):
        if len(data) < 2:
            raise ValueError("Data should have [images, targets]")
        
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in data[1]]
        imgs = list(image.to(self.device) for image in data[0])
        
        return imgs, targets
    
@task
def train_model_task(train_data_loader, val_data_loader, num_epochs):
    model = ModelTrainer()
    model.train(train_data_loader, val_data_loader, num_epochs)
    return model

def train_model_task_(train_data_loader, val_data_loader, num_epochs):
    model = ModelTrainer()
    model.train(train_data_loader, val_data_loader, num_epochs)
    return model
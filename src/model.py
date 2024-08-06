import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from prefect import task

class ModelTrainer:
    def __init__(self, num_classes=9, learning_rate=0.001, momentum=0.9, weight_decay=0.0005, step_size=3, gamma=0.1):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        
        # Initialize the model, optimizer, and learning rate scheduler
        self.model = self._build_model()
        self.device = self.get_device()
        self.model.to(self.device)
        self.optimizer = self._build_optimizer(self.model, learning_rate, momentum, weight_decay)
        self.lr_scheduler = self._build_lr_scheduler(self.optimizer, step_size, gamma)

    def _build_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model

    def get_device(self):
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
            for data in train_data_loader:
                imgs, targets = self._build_model_input(data)
                if imgs is None:
                    continue
                
                loss_dict = self.model(imgs, targets)
                loss = sum(v for v in loss_dict.values())
                epoch_loss += loss.item()
                print(f"Intermediate epoch loss: {epoch_loss}")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.lr_scheduler.step()
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss}")

            val_loss = self.evaluate(val_data_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}")

    def evaluate(self, data_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in data_loader:
                imgs, targets = self._build_model_input(data)
                if imgs is None:
                    continue
                
                loss_dict = self.model(imgs, targets)
                loss = sum(v for v in loss_dict.values())
                val_loss += loss.item()
        
        return val_loss / len(data_loader)
    
    def _build_model_input(self, data):
        if len(data) < 2:
            raise ValueError("Data should have [images, targets]")
        
        targets = [target for target in data[1]]
        imgs = [img.to(self.device) for img in data[0]]
        
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
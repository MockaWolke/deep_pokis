from typing import Any
from lightning import LightningModule
from data_loading import PokemonDataset, DataDownloader
import pandas as pd
import torchmetrics
from torch import nn
from abc import ABC, abstractmethod
import torch
import timm

class ModelTemplate(ABC, nn.Module):

    def __init__(self, include_crops, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.include_crops = include_crops
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x, label = None, label_count = None):
        
        pass

class ExampleCNN(ModelTemplate):

    def __init__(self, include_crops, num_classes, imgsz, *args, **kwargs) -> None:
        super().__init__(include_crops, num_classes, *args, **kwargs)

        self.imgsz = imgsz
        
        
        
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(6 if include_crops else 3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3,padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3,padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3,padding=1)
        
        self.head = nn.Linear(128, self.num_classes)
        



    def forward(self, x, label=None, label_count=None):
        
        
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool2(x)
        x = nn.functional.relu(self.conv4(x))
        
        x = x.mean(axis = (2,3))
        return self.head(x)
        
class ExampleMlp(ModelTemplate):

    def __init__(self, include_crops, num_classes, imgsz, *args, **kwargs) -> None:
        super().__init__(include_crops, num_classes, *args, **kwargs)

        self.imgsz = imgsz
        self.pool_factor = 1
        self.pool = nn.MaxPool2d(self.pool_factor)
        self.lin1 = nn.Linear(
            (
                6 * (imgsz // self.pool_factor) ** 2
                if include_crops
                else 3 * (imgsz // self.pool_factor) ** 2
            ),
            64,
        )
        self.flatten = nn.Flatten()
        self.lin2 = nn.Linear(64, 256)
        self.lin3 = nn.Linear(256, self.num_classes)

    def forward(self, x, label=None, label_count=None):

        x = self.flatten(self.pool(x))
        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin2(x))
        return self.lin3(x)

        
class TimmModel(ModelTemplate):

    def __init__(
        self, include_crops, num_classes, backbone_name = "convnextv2_tiny", dropout=0.0, *args, **kwargs
    ) -> None:
        super().__init__(include_crops, num_classes, *args, **kwargs)

        self.backbone_name = backbone_name

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            in_chans=6 if include_crops else 3,
            features_only=True,
            out_indices=(-1,),
        )
        self.dropout = nn.Dropout(dropout)

        self.emb_dim = self.compute_embedding(
            torch.zeros(1, 6 if include_crops else 3, 256, 256)
        ).shape[-1]
        
        self.norm = nn.LayerNorm(self.emb_dim)

        self.head = nn.Linear(self.emb_dim, num_classes)

    def compute_embedding(self, x):

        x = self.backbone(x)[0]
        x = x.mean((2, 3))
        return x
    
    def forward(self, x,  label=None, label_count=None):
        
        x = self.compute_embedding(x)
        
        x = self.norm(self.dropout(x))
        
        return self.head(x)
        

class LightningWrapper(LightningModule):
    
    def __init__(self, model,) -> None:
        super().__init__()
        
        self.model = model
        

        self.num_classes = self.model.num_classes
        
        
        self.metrics = nn.ModuleDict({
            "acc":torchmetrics.Accuracy(task="multiclass", num_classes = self.num_classes),
            "f1_score" : torchmetrics.F1Score(task="multiclass", num_classes = self.num_classes),
            "precision": torchmetrics.Precision(task="multiclass", num_classes = self.num_classes),
            "recall" : torchmetrics.Recall(task="multiclass", num_classes = self.num_classes),
            })
        self.loss_metric = torchmetrics.MeanMetric()
        
        self.loss_func = nn.CrossEntropyLoss()
        
        
    def forward(self, x, label = None, label_count = None):
        
        return self.model(x, label = label, label_count = label_count)
        
    def training_step(self, data, index):
        
        if len(data)== 2:
            x, y = data
        
            pred = self.forward(x, y,)
            
        elif len(data) ==3:
            x, y, class_counts = data
            pred = self.forward(x, y, class_counts)
            
        
        
        loss = self.loss_func(pred, y)
        
        loss_mean = self.loss_metric(loss)
        self.log("loss", loss_mean, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for name, metric in self.metrics.items():
            
            value = metric(pred, y)
            self.log(name, value, on_step= name == "acc", on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, data, index):
        
        if len(data)== 2:
            x, y = data
        
            pred = self.forward(x, y,)
            
        elif len(data) ==3:
            x, y, class_counts = data
            pred = self.forward(x, y, class_counts)
        
        pred = self.forward(x)
        
        loss = self.loss_func(pred, y)


        loss_mean = self.loss_metric(loss)
        self.log("val_loss", loss_mean, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        for name, metric in self.metrics.items():
            
            value = metric(pred, y)
            self.log("val_" + name, value, on_step=False, on_epoch=True, prog_bar=True, logger=True)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'val_loss',  # Optional: specify what metric to monitor
            }
        }
        
        
        

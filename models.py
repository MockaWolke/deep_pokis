from typing import Any
from lightning import LightningModule
from data_loading import PokemonDataset, DataDownloader
import pandas as pd
import torchmetrics
from torch import nn
from abc import ABC, abstractmethod
import torch
import timm
from torchmetrics.functional import accuracy, precision, recall, f1_score
from collections import defaultdict


class ModelTemplate(ABC, nn.Module):

    def __init__(self, include_crops, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.include_crops = include_crops
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x, label=None, label_count=None):

        pass


class ExampleCNN(ModelTemplate):

    def __init__(self, include_crops, num_classes, imgsz, *args, **kwargs) -> None:
        super().__init__(include_crops, num_classes, *args, **kwargs)

        self.imgsz = imgsz

        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(
            6 if include_crops else 3, 16, kernel_size=5, stride=2, padding=2
        )
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.head = nn.Linear(128, self.num_classes)

    def forward(self, x, label=None, label_count=None):

        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool2(x)
        x = nn.functional.relu(self.conv4(x))

        x = x.mean(axis=(2, 3))
        return self.head(x)


class ExampleMlp(ModelTemplate):

    def __init__(self, include_crops, num_classes, imgsz, *args, **kwargs) -> None:
        super().__init__(include_crops, num_classes, *args, **kwargs)

        self.imgsz = imgsz
        self.pool_factor = 2
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
        self.lin3 = nn.Linear(256, 256)
        self.lin4 = nn.Linear(256, self.num_classes)

    def forward(self, x, label=None, label_count=None):

        x = self.flatten(self.pool(x))
        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin2(x))
        x = nn.functional.relu(self.lin3(x))
        return self.lin4(x)


class TimmModel(ModelTemplate):

    def __init__(
        self,
        include_crops,
        num_classes,
        backbone_name="convnextv2_tiny",
        dropout=0.0,
        *args,
        **kwargs,
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

    def forward(self, x, label=None, label_count=None):

        x = self.compute_embedding(x)

        x = self.norm(self.dropout(x))

        return self.head(x)


class LightningWrapper(LightningModule):

    def __init__(self, model, metric_average="micro") -> None:
        super().__init__()

        self.model = model

        self.num_classes = self.model.num_classes

        self.metrics = nn.ModuleDict(
            {
                "acc": torchmetrics.Accuracy(
                    task="multiclass", num_classes=self.num_classes
                ),
                "f1_score": torchmetrics.F1Score(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=metric_average,
                ),
                "precision": torchmetrics.Precision(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=metric_average,
                ),
                "recall": torchmetrics.Recall(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=metric_average,
                ),
            }
        )
        self.loss_metric = torchmetrics.MeanMetric()

        self.loss_func = nn.CrossEntropyLoss()

        self.test_metrics = nn.ModuleDict(
            {
                "acc": torchmetrics.Accuracy(
                    task="multiclass", num_classes=self.num_classes, average="none"
                ),
                "f1_score": torchmetrics.F1Score(
                    task="multiclass", num_classes=self.num_classes, average="none"
                ),
                "precision": torchmetrics.Precision(
                    task="multiclass", num_classes=self.num_classes, average="none"
                ),
                "recall": torchmetrics.Recall(
                    task="multiclass", num_classes=self.num_classes, average="none"
                ),
            }
        )


        self.id2class = {int(i): v for i, v in DataDownloader.get_id2class().items()}

    def forward(self, x, label=None, label_count=None):

        return self.model(x, label=label, label_count=label_count)

    def training_step(self, data, index):

        if len(data) == 2:
            x, y = data

            pred = self.forward(
                x,
                y,
            )

        elif len(data) == 3:
            x, y, class_counts = data
            pred = self.forward(x, y, class_counts)

        loss = self.loss_func(pred, y)

        loss_mean = self.loss_metric(loss)
        self.log(
            "loss", loss_mean, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        for name, metric in self.metrics.items():

            value = metric(pred, y)
            self.log(
                name,
                value,
                on_step=name == "acc",
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return loss

    def validation_step(self, data, index):

        if len(data) == 2:
            x, y = data

            pred = self.forward(
                x,
                y,
            )

        elif len(data) == 3:
            x, y, class_counts = data
            pred = self.forward(x, y, class_counts)

        pred = self.forward(x)

        loss = self.loss_func(pred, y)

        loss_mean = self.loss_metric(loss)
        self.log(
            "val_loss",
            loss_mean,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        for name, metric in self.metrics.items():

            value = metric(pred, y)
            self.log(
                "val_" + name,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def test_step(self, data, index):

        if len(data) == 2:
            x, y = data

            pred = self.forward(
                x,
                y,
            )

        elif len(data) == 3:
            x, y, class_counts = data
            pred = self.forward(x, y, class_counts)

        pred = self.forward(x)

        for metric in self.test_metrics.values():
            metric(pred, y)

    def on_test_epoch_end(self) -> None:
        results = {
            name: metric.compute().cpu().numpy()
            for name, metric in self.test_metrics.items()
        }
        
        df = {}
        for metric, values in results.items():
            
            df[metric] = [values[index] for index, cl in self.id2class.items()]
            
        self.test_results = pd.DataFrame(df, index=[cl for index, cl in self.id2class.items()])
            
        

    def configure_optimizers(self, lr=1e-3, sheduler_kwgs={"tmax"}):

        self.optim = torch.optim.Adam(self.parameters(), 1e-3)

        return {
            "optimizer": self.optim,
        }

from models import ModelTemplate, LightningWrapper, TimmModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loading import PokemonDataset
import os
from lightning import Trainer
from datetime import datetime
import timm
from lightning.pytorch.loggers import WandbLogger
from torch import nn
import torch
import torchvision.transforms.v2 as transforms
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import io
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image

parser = argparse.ArgumentParser(
    description="Train a Pokemon classifier using Timm models."
)
parser.add_argument(
    "--cores", type=int, default=os.cpu_count(), help="Number of CPU cores to use"
)
parser.add_argument("--imgsz", type=int, default=224, help="Image size")
parser.add_argument("--crop_mode", type=str, default="crop", help="Cropping mode")
parser.add_argument("--bs", type=int, default=64, help="Batch size")
parser.add_argument(
    "--backbone", type=str, default="mobilenetv3_small_050", help="Model backbone"
)
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
parser.add_argument(
    "--name",
    type=str,
    default=None,
)
parser.add_argument("--cat_model", type=int, default=0)
parser.add_argument("--n_pred_imgs", type=int, default=10)
parser.add_argument("--stop_patience", type=int, default=None)


args = parser.parse_args()

if args.name is None:
    args.name = f"{args.backbone}_{args.crop_mode}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"


transform_pipeline = transforms.Compose(
    [
        transforms.ToDtype(torch.float32),
        transforms.Normalize(mean=(0,) * 3, std=(255,) * 3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.5),  # 50% chance of vertical flip
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    ]
)


train_dataset = PokemonDataset(
    "train", args.crop_mode, imgsz=args.imgsz, transforms=transform_pipeline
)
val_dataset = PokemonDataset("val", args.crop_mode, imgsz=args.imgsz)
train_loader = DataLoader(
    train_dataset, batch_size=args.bs, num_workers=args.cores, shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=args.bs, num_workers=args.cores)


class CatModel(ModelTemplate):

    def __init__(
        self,
        include_crops,
        num_classes,
        backbone_name="mobilenetv3_small_075.lamb_in1k",
        dropout=0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(include_crops, num_classes, *args, **kwargs)

        self.backbone = timm.create_model(
            backbone_name,
            True,
            num_classes=0,
        )

        self.dropout = nn.Dropout(dropout)
        self.emb_dims = self.backbone(torch.randn(1, 3, 256, 266)).squeeze().shape[0]

        if self.include_crops:
            self.emb_dims *= 2

        self.head = nn.Linear(self.emb_dims, num_classes)

    def forward(self, x, label=None, label_count=None):

        if x.shape[1] == 3:
            x = self.backbone(x)
        elif x.shape[1] == 6:
            x1 = self.backbone(x[:, :3])
            x2 = self.backbone(x[:, 3:])
            x = torch.cat((x1, x2), -1)

        x = self.dropout(x)

        return self.head(x)


class TimmModel(ModelTemplate):

    def __init__(
        self,
        include_crops,
        num_classes,
        backbone_name="mobilenetv3_small_075.lamb_in1k",
        dropout=0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(include_crops, num_classes, *args, **kwargs)

        self.backbone = timm.create_model(
            backbone_name,
            True,
            num_classes=0,
            in_chans=6 if include_crops else 3,
        )
        self.dropout = nn.Dropout(dropout)
        self.emb_dims = (
            self.backbone(torch.randn(1, 6 if include_crops else 3, 256, 266))
            .squeeze()
            .shape[0]
        )
        self.head = nn.Linear(self.emb_dims, num_classes)

    def forward(self, x, label=None, label_count=None):

        x = self.backbone(x)
        x = self.dropout(x)

        return self.head(x)


if args.cat_model:
    model = CatModel(args.crop_mode == "both", 18, args.backbone, dropout=args.dropout)
else:
    model = TimmModel(args.crop_mode == "both", 18, args.backbone, dropout=args.dropout)


wandb_logger = WandbLogger(
    args.name,
    dir="logs",
    log_model=True,
    config=vars(args),
    save_code=True,
)

callbacks = []

if args.stop_patience:
    callbacks.append(
        EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=args.stop_patience,
            verbose=True,
        )
    )
    
checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    dirpath='checkpoints/',
    filename='best-checkpoint',
    save_top_k=1,
    mode='max',
)
callbacks.append(checkpoint_callback)



wrapper = LightningWrapper(model, metric_average="micro")
trainer = Trainer(
    devices="auto", max_epochs=args.epochs, logger=wandb_logger, callbacks=callbacks
)
trainer.fit(wrapper, train_loader, val_loader)

if checkpoint_callback.best_model_path:
    wrapper = wrapper.load_from_checkpoint(checkpoint_callback.best_model_path)

wrapper.load_from_checkpoint(checkpoint_callback.best_model_path)

eval_val = PokemonDataset("val", args.crop_mode, imgsz=args.imgsz, synth_frac=0)
eval_loader = DataLoader(eval_val, batch_size=args.bs, num_workers=args.cores)

trainer.test(wrapper, eval_loader)

if hasattr(wrapper, "test_results"):
    wandb_logger.log_table("clean_val_by_class", dataframe=wrapper.test_results)


def compute_preds(trainer, wrapper):
    print("computing preds")
    test_dataset = PokemonDataset("test", args.crop_mode, imgsz=args.imgsz)

    test_loader = DataLoader(
        test_dataset, batch_size=args.bs, num_workers=args.cores, shuffle=False
    )

    preds = trainer.predict(wrapper, test_loader)

    ids = np.concatenate([np.argmax(i.detach().cpu().numpy(), -1) for i in preds])

    test_dataset.df["main_type"] = ids.astype(str)
    test_dataset.df["Id"] = ids

    return test_dataset.df.copy()


pred_df = compute_preds(trainer, wrapper)

stripped = pred_df[["Id", "main_type"]]

wandb_logger.log_table("submission", dataframe=stripped)


def plot_test_preds(df, id2class) -> Image:
    fig = plt.figure(figsize=(4, 4))
    row = df.query("cropp_exists").sample(1).iloc[0]
    plt.subplot(2, 2, 1)
    plt.imshow(plt.imread(row.path))
    plt.axis("off")
    plt.title(str(row.main_type) + " " + id2class[row.main_type])
    plt.subplot(2, 2, 2)
    plt.axis("off")

    plt.imshow(plt.imread(row.cropped_path))
    plt.axis("off")

    row = df.query("cropp_exists").sample(1).iloc[0]
    plt.subplot(2, 2, 3)
    plt.imshow(plt.imread(row.path))
    plt.axis("off")
    plt.title(str(row.main_type) + " " + id2class[row.main_type])

    plt.subplot(2, 2, 4)
    plt.imshow(plt.imread(row.cropped_path))
    plt.axis("off")
    plt.imshow(plt.imread(row.cropped_path))

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)


try:

    imgs = [plot_test_preds(pred_df, wrapper.id2class) for i in range(args.n_pred_imgs)]

    wandb_logger.log_image("test_pred_imgs", imgs)

except Exception as e:
    print(f"There was an image loggin error: {e}")

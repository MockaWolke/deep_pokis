import math
import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
import torchvision.transforms.v2 as transforms
from data_loading import PokemonDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import io


def gen_transforms(level: str):

    if level == "mild":

        return transforms.Compose(
            [
                transforms.ToDtype(torch.float32),
                transforms.Normalize(mean=(0,) * 3, std=(255,) * 3),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.5),  # 50% chance of vertical flip
                transforms.RandomRotation(degrees=10),
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            ]
        )
    if level == "medium":
        return transforms.Compose(
            [
                transforms.ToDtype(torch.float32),
                transforms.Normalize(mean=(0,) * 3, std=(255,) * 3),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(size=224, scale=(0.85, 0.95)),
                transforms.RandomAffine(
                    degrees=0, translate=(0.05, 0.05), scale=None, shear=5
                ),
                # Optionally, add more transformations here for "medium" level
            ]
        )

    if level == "strong":
        return transforms.Compose(
            [
                transforms.ToDtype(torch.float32),
                transforms.Normalize(mean=(0,) * 3, std=(255,) * 3),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=20),
                transforms.RandomResizedCrop(size=224, scale=(0.7, 0.9)),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            ]
        )


def get_test_df(trainer, wrapper, batchsize, num_workers, crop_mode) -> pd.DataFrame:
    
    dataset = PokemonDataset("test", crop_mode=crop_mode, synth_frac=0)
    loader = DataLoader(dataset, batchsize, num_workers=num_workers)
    preds = trainer.predict(wrapper, loader)
    ids = np.concatenate([np.argmax(i.detach().cpu().numpy(), -1) for i in preds])
    
    dataset.df["main_type"] = ids.astype(str)
    dataset.df["Id"] = ids

    return dataset.df.copy()


# def get_val_umap(trainer, wrapper, batchsize, num_workers, crop_mode) -> Image:
    
#     preds = get_predictions(trainer,wrapper, loader)
    
#     reducer = umap.UMAP()
#     reducer.fit(preds)
#     embedding = reducer.transform(preds)
#     df = dataset.df
#     df["x"] = embedding[:, 0]
#     df["y"] = embedding[:, 1]
    
#     fig = plt.figure(figsize=(6,6))
#     sns.scatterplot(df, x = "x", y="y", hue="main_type")
#     plt.title("2d Embeddings")
#     plt.tight_layout()
#     buf = io.BytesIO()
#     fig.savefig(buf)
#     buf.seek(0)
#     return Image.open(buf)


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

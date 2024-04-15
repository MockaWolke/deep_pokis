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
from tqdm import tqdm
import timm


def extract_mean_and_std(timm_backbone):


    try:

        data_config = timm.data.resolve_model_data_config(timm_backbone)
        mean, std = data_config["mean"], data_config["std"]

    except:
        print("Could not extract dataconfig mean and std, falling back on standard")

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
    return mean, std


def get_val_transform( mean, std):
    return transforms.Normalize(mean=mean, std=std)

def gen_transforms(level: str, mean, std):
    

    if level == "mild":

        return transforms.Compose(
            [
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.5),  # 50% chance of vertical flip
                transforms.RandomRotation(degrees=10),
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            ]
        )
    if level == "medium":
        return transforms.Compose(
            [
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(size=224, scale=(0.85, 0.95)),
                transforms.RandomAffine(
                    degrees=0, translate=(0.05, 0.05), scale=None, shear=5
                ),
                # Optionally, add more transformations here for "medium" level
            ]
        )

    if level == "medium_strong_crop":
        return transforms.Compose(
            [
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(size=224, scale=(0.3, 0.8)),
                transforms.RandomAffine(
                    degrees=0, translate=(0.05, 0.05), scale=None, shear=5
                ),
                # Optionally, add more transformations here for "medium" level
            ]
        )

    if level == "strong":
        return transforms.Compose(
            [
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=20),
                transforms.RandomResizedCrop(size=224, scale=(0.2, 0.8)),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            ]
        )

    if level == "extra_strong":

        return transforms.Compose(
            [
                transforms.Normalize(mean=mean, std=std),  # Common normalization values
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=30),
                transforms.RandomResizedCrop(size=224, scale=(0.2, 0.8)),
                transforms.RandomAffine(
                    degrees=15, translate=(0.2, 0.2), scale=(0.5, 1.5), shear=20
                ),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.7),
                transforms.RandomGrayscale(
                    p=0.2
                ),  # Convert image to grayscale with a probability
            ]
        )

    raise ValueError()


def get_test_df(trainer, wrapper, dataset, loader) -> pd.DataFrame:
    preds = trainer.predict(wrapper, loader)
    ids = np.concatenate([np.argmax(i.detach().cpu().numpy(), -1) for i in preds])

    dataset.df["main_type"] = ids.astype(str)
    dataset.df["Id"] = dataset.df.image_id

    return dataset.df.copy()


def plot_umap(embeddings, labels, id2class, title=None) -> Image:

    reducer = umap.UMAP()
    reducer.fit(embeddings)
    twodim = reducer.transform(embeddings)

    df = pd.DataFrame({"x": twodim[:, 0], "y": twodim[:, 1], "main_type": labels})
    df.main_type = df.main_type.map(id2class)
    fig = plt.figure(figsize=(6, 6))
    sns.scatterplot(df, x="x", y="y", hue="main_type")

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1))

    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)


def get_top_10_best(combined_embeddings, test_embeddings, combined_label, test_dataset):
    combined_embeddings_norm = torch.nn.functional.normalize(
        torch.tensor(combined_embeddings)
    )
    test_embeddings_norm = torch.nn.functional.normalize(torch.tensor(test_embeddings))

    cos_scores = test_embeddings_norm @ combined_embeddings_norm.T
    indices = torch.topk(cos_scores, k=10, axis=1).indices.cpu().numpy()

    top_10_labels = combined_label[indices]
    top_10_df = pd.DataFrame({f"top_{i}": top_10_labels[:, i] for i in range(10)})
    top_10_df["Id"] = test_dataset.df.image_id.values.copy()
    return top_10_df


def get_embeddings(arcface, loader):

    embeddings, labels = [], []
    arcface.eval()
    for data in tqdm(loader):

        if torch.is_tensor(data):
            x = data
        else:
            x, y, _ = arcface.unpack_data(data)
            labels.append(y.cpu().numpy())

        embeddings.append(
            arcface.model.get_embedding(x.to(arcface.device)).detach().cpu().numpy()
        )
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels) if labels else np.array([])

    return embeddings, labels


def plot_test_preds(df, id2class) -> Image:
    fig = plt.figure(figsize=(4, 4))
    row = df.query("cropp_exists").sample(1).iloc[0]
    plt.subplot(2, 2, 1)
    plt.imshow(plt.imread(row.path))
    plt.axis("off")
    plt.title(str(row.main_type) + " " + id2class[int(row.main_type)])
    plt.subplot(2, 2, 2)

    plt.imshow(plt.imread(row.cropped_path))
    plt.axis("off")

    row = df.query("cropp_exists").sample(1).iloc[0]
    plt.subplot(2, 2, 3)
    plt.imshow(plt.imread(row.path))
    plt.axis("off")
    plt.title(str(row.main_type) + " " + id2class[int(row.main_type)])

    plt.subplot(2, 2, 4)
    plt.imshow(plt.imread(row.cropped_path))
    plt.axis("off")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)

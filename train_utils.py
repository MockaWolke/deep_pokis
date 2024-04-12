import math
import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
import torchvision.transforms.v2 as transforms


def gen_cos_anneal_warmup(optim, warmup_steps, anneal_steps, min_cos_lr):

    return SequentialLR(
        optimizer=optim,
        schedulers=[
            LinearLR(optim, 1e-7, 1, warmup_steps),
            CosineAnnealingLR(optim, anneal_steps, eta_min=min_cos_lr),
        ],
    )


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

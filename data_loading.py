import os
import shutil
from torch.utils.data import Dataset
import zipfile
import pandas as pd
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datadownloader import DataDownloader
import torchvision.transforms.v2 as T

standard_norm_transf = T.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225))
class RevisedDataset(Dataset):

    def __init__(
        self,
        df : pd.DataFrame,
        mode : str,
        crop_mode : str,
        transforms=standard_norm_transf,
        imgsz=256,
    ) -> None:
        super().__init__()

        if mode not in [
            "train",
            "val",
            "test",
        ]:
            raise ValueError()
        
        self.mode = mode

        assert crop_mode == "crop"
        self.df: pd.DataFrame = df.copy()

        self.resize = torchvision.transforms.Resize((imgsz, imgsz), antialias=True)

        self.num_classes = len(self.df.class_id.dropna().unique())
        
        self.transforms = transforms

    def __len__(self):

        return len(self.df)

    def plot_examples(self, n=4):

        fig = plt.figure(figsize=(6, 6))

        for x in range(n):
            for y in range(n):
                index = x + 4 * y + 1
                plt.subplot(n, n, index)

                row = self.df.sample(1).iloc[0]

                plt.imshow(plt.imread(row.path))

                plt.axis("off")

                plt.title(f"{row.main_type} - {int(row.class_id)}")
        plt.tight_layout()
        plt.show()

    def plot_examples_crops(self, n=4):

        fig = plt.figure(figsize=(6, 6))

        for x in range(n):
            for y in range(n):
                index = x + 4 * y + 1
                plt.subplot(n, n, index)

                row = self.df.query("cropp_exists").sample(1).iloc[0]

                plt.imshow(plt.imread(row.cropped_path))

                plt.axis("off")

                plt.title(f"{row.main_type} - {int(row.class_id)}")
        plt.tight_layout()
        plt.show()

    def plot_examples_both(
        self,
    ):

        fig = plt.figure(figsize=(6, 6))
        row = self.df.query("cropp_exists").sample(1).iloc[0]
        plt.subplot(2, 2, 1)
        plt.imshow(plt.imread(row.path))
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.axis("off")
        plt.title(f"{row.main_type} - {int(row.class_id)}")
        plt.imshow(plt.imread(row.cropped_path))
        plt.axis("off")

        row = self.df.query("cropp_exists").sample(1).iloc[0]
        plt.subplot(2, 2, 3)
        plt.imshow(plt.imread(row.path))
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.imshow(plt.imread(row.cropped_path))
        plt.axis("off")
        plt.imshow(plt.imread(row.cropped_path))

        plt.tight_layout()
        plt.show()

    def __getitem__(self, index) -> tuple[torch.TensorType, int, int]:

        row = self.df.iloc[index]




        img = (
            self.resize(torchvision.io.read_image(row.cropped_path))
            if row.cropp_exists
            else self.resize(torchvision.io.read_image(row.path))
        )
        img = self.transforms(img.float() / 255)

        if self.mode == "test":
            return img

        return img, int(row.class_id), int(row.class_counts)

    def viz_transorms(
        self,
        img_index=None,
        crop=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):

        if img_index is None:
            img_index = np.random.choice(len(self.df))

        row = self.df.iloc[img_index]

        if crop:

            img = (
                self.resize(torchvision.io.read_image(row.cropped_path))
                if row.cropp_exists
                else self.resize(torchvision.io.read_image(row.path))
            )

        else:

            img = self.resize(torchvision.io.read_image(row.path))

        imgs = [self.transforms(img.float() / 255) for _ in range(16)]

        fig, axes = plt.subplots(4, 4, figsize=(4, 4))
        for i, ax in enumerate(axes.flat):

            img = imgs[i].permute(1, 2, 0).numpy()
            img_normalized = img * np.array(std)[None,None,:] + np.array(mean)[None,None,:]
            img_normalized = np.clip(img_normalized, 0, 1)
            ax.imshow(img_normalized)
            ax.axis("off")
        plt.tight_layout()
        plt.show()



class PokemonDataset(Dataset):

    def __init__(
        self,
        mode,
        crop_mode: str = "both",
        synth_frac=1.0,
        transforms=standard_norm_transf,
        imgsz=256,
    ) -> None:
        super().__init__()

        if mode not in [
            "train",
            "val",
            "test",
        ]:
            raise ValueError()

        if crop_mode not in ["img", "both", "crop"]:
            raise ValueError()

        self.mode = mode
        self.crop_mode = crop_mode
        self.transforms = transforms
        self.imgsz = imgsz

        if not DataDownloader.complete():
            DataDownloader.download()

        assert DataDownloader.complete(), "Data is missing"

        orig_df = DataDownloader.get_info_csv()

        self.df: pd.DataFrame = orig_df.query("ds_type == @mode").copy()

        aug_string = f"aug_{mode}"

        self.df = pd.concat(
            (
                self.df,
                orig_df.query("ds_type == @aug_string").copy().sample(frac=synth_frac),
            )
        )

        self.resize = torchvision.transforms.Resize((imgsz, imgsz), antialias=True)

        self.num_classes = len(self.df.class_id.dropna().unique())

    def __len__(self):

        return len(self.df)

    def plot_examples(self, n=4):

        fig = plt.figure(figsize=(6, 6))

        for x in range(n):
            for y in range(n):
                index = x + 4 * y + 1
                plt.subplot(n, n, index)

                row = self.df.sample(1).iloc[0]

                plt.imshow(plt.imread(row.path))

                plt.axis("off")

                plt.title(f"{row.main_type} - {int(row.class_id)}")
        plt.tight_layout()
        plt.show()

    def plot_examples_crops(self, n=4):

        fig = plt.figure(figsize=(6, 6))

        for x in range(n):
            for y in range(n):
                index = x + 4 * y + 1
                plt.subplot(n, n, index)

                row = self.df.query("cropp_exists").sample(1).iloc[0]

                plt.imshow(plt.imread(row.cropped_path))

                plt.axis("off")

                plt.title(f"{row.main_type} - {int(row.class_id)}")
        plt.tight_layout()
        plt.show()

    def plot_examples_both(
        self,
    ):

        fig = plt.figure(figsize=(6, 6))
        row = self.df.query("cropp_exists").sample(1).iloc[0]
        plt.subplot(2, 2, 1)
        plt.imshow(plt.imread(row.path))
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.axis("off")
        plt.title(f"{row.main_type} - {int(row.class_id)}")
        plt.imshow(plt.imread(row.cropped_path))
        plt.axis("off")

        row = self.df.query("cropp_exists").sample(1).iloc[0]
        plt.subplot(2, 2, 3)
        plt.imshow(plt.imread(row.path))
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.imshow(plt.imread(row.cropped_path))
        plt.axis("off")
        plt.imshow(plt.imread(row.cropped_path))

        plt.tight_layout()
        plt.show()

    def __getitem__(self, index) -> tuple[torch.TensorType, int, int]:

        row = self.df.iloc[index]

        if self.crop_mode == "img":

            img = self.resize(torchvision.io.read_image(row.path))
            img = self.transforms(img.float() / 255)

        elif self.crop_mode == "crop":

            img = (
                self.resize(torchvision.io.read_image(row.cropped_path))
                if row.cropp_exists
                else self.resize(torchvision.io.read_image(row.path))
            )
            img = self.transforms(img.float() / 255)

        elif self.crop_mode == "both":

            img = self.resize(torchvision.io.read_image(row.path))
            img = self.transforms(img.float() / 255)

            crop = (
                self.resize(torchvision.io.read_image(row.cropped_path))
                if row.cropp_exists
                else torch.zeros_like(img)
            )
            crop = self.transforms(crop.float() / 255)

            img = torch.cat((img, crop), 0)

        if self.mode == "test":
            return img

        return img, int(row.class_id), int(row.class_counts)

    def viz_transorms(
        self,
        img_index=None,
        crop=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):

        if img_index is None:
            img_index = np.random.choice(len(self.df))

        row = self.df.iloc[img_index]

        if crop:

            img = (
                self.resize(torchvision.io.read_image(row.cropped_path))
                if row.cropp_exists
                else self.resize(torchvision.io.read_image(row.path))
            )

        else:

            img = self.resize(torchvision.io.read_image(row.path))

        imgs = [self.transforms(img.float() / 255) for _ in range(16)]

        fig, axes = plt.subplots(4, 4, figsize=(4, 4))
        for i, ax in enumerate(axes.flat):

            img = imgs[i].permute(1, 2, 0).numpy()
            img_normalized = img * np.array(std)[None,None,:] + np.array(mean)[None,None,:]
            img_normalized = np.clip(img_normalized, 0, 1)
            ax.imshow(img_normalized)
            ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    # DataDownloader.create_zip()
    DataDownloader.create_zip()
    # dataset = PokemonDataset("val", 12, )
    # print(dataset[0])

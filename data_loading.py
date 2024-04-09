import os
import shutil
from torch.utils.data import Dataset
import zipfile
import pandas as pd
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt


class DataDownloader:

    needed_paths = [
        "info.csv",
        "train_cropped",
        "train",
        "test_cropped",
        "test",
    ]

    prefix = "data"
    share_path = "https://drive.google.com/file/d/1IGX5we-xkP3mzk7cGkB7LNXaY1kvl3w0/view?usp=sharing"

    @classmethod
    def get_paths(cls):
        return [os.path.join(cls.prefix, p) for p in cls.needed_paths]

    @classmethod
    def download(cls):
        import gdown

        print("Downloading Dataset")
        os.makedirs(cls.prefix, exist_ok=True)

        output_path = os.path.join(cls.prefix, "dataset.zip")
        gdown.download(url=cls.share_path, output=output_path, quiet=False, fuzzy=True)

        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(cls.prefix)

        os.remove(output_path)

    @classmethod
    def get_info_csv(cls):
        return pd.read_csv(
            os.path.join(
                cls.prefix,
                "info.csv",
            ),
            index_col=0,
        )

    @classmethod
    def complete(cls):

        for path in cls.get_paths():
            if not os.path.exists(path):
                return False

        return True

    @classmethod
    def create_zip(cls):

        assert cls.complete()

        zip_path = os.path.join(cls.prefix, "pokemon_dataset.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for path in cls.get_paths():
                if os.path.isdir(path):
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            zipf.write(
                                os.path.join(root, file),
                                os.path.relpath(
                                    os.path.join(root, file), os.path.join(path, "..")
                                ),
                            )
                else:
                    zipf.write(path, os.path.basename(path))
        # crate pokemon_dataset.zip with all files


class PokemonDataset(Dataset):

    def __init__(
        self, mode, crop_mode: str = "both", transforms=lambda x: x, imgsz=256
    ) -> None:
        super().__init__()

        if mode not in ["train", "val", "test"]:
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

        self.df: pd.DataFrame = (
            DataDownloader.get_info_csv().query("ds_type == @mode").copy()
        )

        self.resize = torchvision.transforms.Resize((imgsz, imgsz), antialias = True)

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

        elif self.crop_mode == "crop":

            img = (
                self.resize(torchvision.io.read_image(row.cropped_path))
                if row.cropp_exists
                else self.resize(torchvision.io.read_image(row.path))
            )

        elif self.crop_mode == "both":

            img = self.resize(torchvision.io.read_image(row.path))
            crop = (
                self.resize(torchvision.io.read_image(row.cropped_path))
                if row.cropp_exists
                else torch.zeros_like(img)
            )

            img = torch.cat((img, crop), 0)

        img = img.float() / 255

        img = self.transforms(img)

        if self.mode == "test":
            return img

        return img, int(row.class_id), int(row.class_counts)


if __name__ == "__main__":

    # DataDownloader.create_zip()
    DataDownloader.create_zip()
    # dataset = PokemonDataset("val", 12, )
    # print(dataset[0])

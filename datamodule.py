import os
import pandas as pd
from datadownloader import DataDownloader
from lightning import LightningDataModule
from data_loading import RevisedDataset, standard_norm_transf
from torch.utils.data import DataLoader


class PokemonDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        val_og_percentage=1.0,
        fill_val_to=0.1,
        num_workers=os.cpu_count(),
        pin_memory=False,
        imgsz=224,
        train_transforms=standard_norm_transf,
        val_and_pred_transforms=standard_norm_transf,
    ):
        super().__init__()
        self.batch_size = batch_size

        if not DataDownloader.complete():
            DataDownloader.download()

        assert DataDownloader.complete(), "Data is missing"

        self.df = pd.read_csv("data/info.csv", index_col=0)
        self.val_og_percentage = val_og_percentage
        self.fill_val_to = fill_val_to
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_df, self.val_df = self.create_split()
        self.test_df = self.df.query("ds_type == 'test'").copy()
        self.train_transforms = train_transforms
        self.val_and_pred_transforms = val_and_pred_transforms
        self.imgsz = imgsz

    def get_train_dataset(self):

        return RevisedDataset(
            self.train_df, "train", "crop", self.train_transforms, imgsz=self.imgsz
        )

    def get_val_dataset(self, without_synth=False):

        df = self.val_df if not without_synth else self.val_df.query("ds_type == 'og'")

        return RevisedDataset(
            df, "val", "crop", self.val_and_pred_transforms, imgsz=self.imgsz
        )

    def get_test_dataset(self):

        return RevisedDataset(
            self.test_df, "test", "crop", self.val_and_pred_transforms, imgsz=self.imgsz
        )

    def create_split(self):
        """
        Splits the dataset into training and validation sets based on the type of data (original 'og' or synthesized 'aug').
        
        The function initially allocates all original data ('og') to the validation set based on the specified `val_og_percentage`.
        If the proportion of the validation set is less than `fill_val_to`, additional synthesized data ('aug') is added to the
        validation set until the desired proportion is achieved.
        """
        
        # Exclude test data
        df_without_test = self.df.query("ds_type != 'test'").copy()

        # Get original data indices
        og_index = df_without_test.query("ds_type == 'og'").index.to_list()
        val_index = og_index[: int(len(og_index) * self.val_og_percentage)]

        # Calculate current proportion of validation data
        current_val_proportion = len(val_index) / len(df_without_test)

    # Check if additional data is needed to reach the desired validation fill
        if current_val_proportion < self.fill_val_to:
            required_additional_val = int(len(df_without_test) * self.fill_val_to - len(val_index))
            aug_index = df_without_test.query("ds_type == 'aug'").index.to_list()
            val_index += aug_index[:required_additional_val]

            val_df = df_without_test.loc[val_index].copy()

            train_index = list(set(df_without_test.index.to_list()).difference(val_index))
            train_df = df_without_test.loc[train_index].copy()

            return train_df, val_df

    def train_dataloader(self):
        return DataLoader(
            self.get_train_dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self, without_synth=False):
        return DataLoader(
            self.get_val_dataset(without_synth=without_synth),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.get_test_dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

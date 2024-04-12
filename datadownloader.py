import os
import zipfile
import pandas as pd
import json
import argparse

class DataDownloader:

    needed_paths = [
        "info.csv",
        "train_cropped",
        "train",
        "test_cropped",
        "test",
        "generated_cropped",
        "generated",
        "class2id.json",
        "id2class.json",
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

    @classmethod 
    def get_class2id(cls):
        
        with open("data/class2id.json") as f:
            return json.load(f)
        
    @classmethod 
    def get_id2class(cls):
        
        with open("data/id2class.json") as f:
            return json.load(f)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", type=str, choices=["zip","load"])
    
    cmd = parser.parse_args().cmd
    
    if cmd == "zip":
        DataDownloader.create_zip()
        
    elif cmd == "load":
        DataDownloader.download()
        assert DataDownloader.complete(), "not all data found"
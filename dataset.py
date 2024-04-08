import os
import shutil
from torch.utils.data import Dataset
import zipfile

class DataDownloader:

    needed_paths = ["info.csv", 
                    "train_cropped",                    
                    "train",
                    "test_cropped",                    
                    "test",
                    ]
    
    prefix = "data"
    
    @classmethod
    def get_paths(cls):
        return [os.path.join(cls.prefix, p) for p in cls.needed_paths]
    
    @classmethod
    def assert_complete(cls):
    
        for path in cls.get_paths():
            assert os.path.exists(path)
        
    @classmethod
    def create_zip(cls):
        
        cls.assert_complete()
        
        zip_path = os.path.join(cls.prefix, "pokemon_dataset.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for path in cls.get_paths():
                if os.path.isdir(path):
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            zipf.write(os.path.join(root, file),
                                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))
                else:
                    zipf.write(path, os.path.basename(path))
        print(f"Created zip at {zip_path}")
        # create pokemon_dataset.zip with all files
        

# class PokemonDataset(Dataset):
    

    
#     def __init__(self) -> None:
#         super().__init__()


if __name__ == "__main__":
    
    DataDownloader.create_zip()
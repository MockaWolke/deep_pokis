import pandas as pd
import random 
import numpy as np
from glob import glob
from tqdm import tqdm
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageOps
from multiprocessing import Pool
import random
random.seed(0)
np.random.seed(0)


N_TOTAL = 5000


background_image = glob("data/places_imgs/**/*.jpg", recursive=True)

for i, path in enumerate(tqdm(np.random.choice(background_image, N_TOTAL))):
    save_path = f"data/generated/background_{i}.jpg"

    
    img = Image.open(path).convert("RGB")
    og_width, _ = img.size
    new_width = min(og_width, 400) # make to 4/3 format
    new_height = int(new_width / 4 * 3)
    
    img = img.resize((new_width, new_height))
    
    img.save(save_path)

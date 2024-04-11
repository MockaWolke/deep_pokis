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
import hashlib
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
import random
random.seed(0)
np.random.seed(0)


IMGSZ = 224
N_TOTAL = 20000
CORES = 7

df = pd.read_csv("data/parsed_from_index.csv", index_col=0).drop_duplicates("name")
df.primary.value_counts()

def sample_even(n_times):
    
    samples = []
    to_class = df.groupby("primary")["name"].apply(lambda x: list(x))
    with_name = df.set_index("name")
    
    for _ in tqdm(range(n_times)):

        index = random.choice(random.choice(to_class.values))
        row = with_name.loc[index]
        samples.append(row)
    return pd.DataFrame(samples)

samples = sample_even(N_TOTAL)
samples.value_counts("primary", normalize=True).plot.bar()
plt.show()
samples["y_center"] = np.random.uniform(0.15, 0.85, len(samples))
samples["x_center"] = np.random.uniform(0.15, 0.85, len(samples))

samples["width"] = np.random.uniform(0.07, 0.20, len(samples))

samples["upper_corner"] = samples.y_center - samples.width/2
samples["left_corner"] = samples.x_center - samples.width/2

background_image = glob("data/places_imgs/*/*/*.jpg")
samples["background_path"] = np.random.choice(background_image, len(samples))

samples["Id"] = np.arange(len(samples))
# samples.to_csv("data/generated_imgs.csv")

samples = pd.read_csv("data/generated_imgs.csv", index_col=0)


def turn_tranparent(img):
    img = img.convert("RGBA")

    pixdata = img.load()

    width, height = img.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y] == (255, 255, 255, 255):
                pixdata[x, y] = (255, 255, 255, 0)
    return img

os.makedirs("data/generated", exist_ok=True)
os.makedirs("data/generated_cropped", exist_ok=True)

def generate_img(row : pd.Series):
    save_path = f"data/generated/{row.Id}.jpg"
    
    if os.path.exists(save_path):
        return
    
    img = Image.open(row.background_path).resize((IMGSZ, IMGSZ)).convert("RGBA")
    back_ground = Image.new("RGBA", img.size)

    pokemon = Image.open(row.image_path)
    pokemon = turn_tranparent(pokemon)

    resize_y = int(1 / (pokemon.size[0] / IMGSZ / row.width) * pokemon.size[0])
    resize_x = int(1 / (pokemon.size[1] / IMGSZ / row.width) * pokemon.size[1])

    pokemon = pokemon.resize((resize_y, resize_x))

    upper_corner = int(row.upper_corner * IMGSZ)
    left_corner = int(row.left_corner * IMGSZ)
    back_ground.paste(pokemon, (upper_corner, left_corner), pokemon.split()[3])
    img = Image.alpha_composite(img, back_ground).convert("RGB")
    img.save(save_path)

rows = [samples.iloc[i] for i in range(len(samples))]

with Pool(CORES) as pool:
        list(tqdm(pool.imap(generate_img, rows), total=len(rows)))
        




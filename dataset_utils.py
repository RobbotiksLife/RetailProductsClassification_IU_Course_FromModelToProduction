import os
import json
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, dataframe, label_map:  dict[str, int], img_dir: str, transform=None):
        self.df = dataframe
        self.label_map = label_map
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_id = self.df.iloc[idx]['ImgId']
        label = self.label_map[self.df.iloc[idx]['categories']]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def prepare_dataset(csv_path: str, img_dir: str, save_label_map_path: str) -> tuple[pd.DataFrame, dict[str, int]]:
    df = pd.read_csv(csv_path)
    df = df[df["ImgId"].apply(lambda x: os.path.isfile(f"{img_dir}/{x}.jpg"))]  # some images don't exist

    categories = sorted(df['categories'].unique())
    label_map = {t: iid for iid, t in enumerate(categories)}
    with open(save_label_map_path, 'w') as f:
        json.dump(label_map, f)

    return df, label_map
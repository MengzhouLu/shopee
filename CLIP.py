import os

import cv2
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import clip
from PIL import Image

import numpy as np
import re
import albumentations
from torch.utils.data import Dataset, DataLoader


def get_transforms(img_size=256):
    return albumentations.Compose([
        albumentations.Resize(img_size, img_size),
        albumentations.Normalize()
    ])
def extract_and_replace_with_standard_unit(tit):
    for cat, units in measurements.items():
        min_unit = units[0][0]  # 获取最小单位
        for unit_name, mult in units:
            pat = fr'\b(\d+(?:[\,\.]\d+)?) ?{unit_name}s?\b'
            tit = re.sub(pat, lambda x: f"{to_num(x.group(1), mult)} {min_unit}", tit)
    return tit.strip()

class LandmarkDataset(Dataset):
    def __init__(self, csv, split, mode, transforms=get_transforms(img_size=256), tokenizer=None):

        self.csv = csv.reset_index()
        self.split = split
        self.mode = mode
        self.transform = transforms
        self.tokenizer = tokenizer

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        text = row.title
        text = extract_and_replace_with_standard_unit(text)

        image = cv2.imread(row.filepath)
        image = image[:, :, ::-1]

        res0 = self.transform(image=image)
        image0 = res0['image'].astype(np.float32)
        image = image0.transpose(2, 0, 1)

        text = self.tokenizer(text, padding='max_length', truncation=True, max_length=16, return_tensors="pt")
        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0]

        if self.mode == 'test':
            return torch.tensor(image), input_ids, attention_mask
        else:
            return torch.tensor(image), input_ids, attention_mask, torch.tensor(row.label_group)

df_sub = pd.read_csv('./test.csv')

df_test = df_sub.copy()
df_test['filepath'] = df_test['image'].apply(lambda x: os.path.join('./', 'test_images', x))

dataset_test = LandmarkDataset(df_test, 'test', 'test', transforms=get_transforms(img_size=256), tokenizer=tokenizer)
test_loader = DataLoader(dataset_test, batch_size=16, num_workers=4)
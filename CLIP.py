import os

import cv2
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import clip
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import nn, optim
from torchvision import transforms, utils
import os
import numpy as np
import re

# parameter
EPOCH = 30
BATCH_SIZE = 128

# load clip model
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training


measurements = {
    'weight': [('mg',1), ('g', 1000), ('gr', 1000), ('gram', 1000), ('kg', 1000000)],
    'length': [('mm',1), ('cm', 10), ('m',1000), ('meter', 1000)],
    'pieces': [ ('pc',1)],
    'memory': [('gb', 1)],
    'volume': [('ml', 1), ('l', 1000), ('liter',1000)]
}
def to_num(x, mult=1):
    x = x.replace(',','.')
    return int(float(x)*mult)
def extract_and_replace_with_standard_unit(tit):
    for cat, units in measurements.items():
        min_unit = units[0][0]  # 获取最小单位
        for unit_name, mult in units:
            pat = fr'\b(\d+(?:[\,\.]\d+)?) ?{unit_name}s?\b'
            tit = re.sub(pat, lambda x: f"{to_num(x.group(1), mult)} {min_unit}", tit)
    return tit.strip()

class LandmarkDataset(Dataset):
    def __init__(self, csv, split, mode):
        super(LandmarkDataset, self).__init__()
        self.tranform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.csv = csv.reset_index()
        self.split = split
        self.mode = mode


    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        text = row.title
        text = extract_and_replace_with_standard_unit(text)

        # image = cv2.imread(row.filepath)
        # image = image[:, :, ::-1]
        image = preprocess(Image.open(row.filepath))
        T = text
        return {'P': image, 'T': T}

        # res0 = self.transform(image=image)
        # image0 = res0['image'].astype(np.float32)
        # image = image0.transpose(2, 0, 1)
        #
        # text = self.tokenizer(text, padding='max_length', truncation=True, max_length=16, return_tensors="pt")
        # input_ids = text['input_ids'][0]
        # attention_mask = text['attention_mask'][0]
        #
        # if self.mode == 'test':
        #     return torch.tensor(image), input_ids, attention_mask
        # else:
        #     return torch.tensor(image), input_ids, attention_mask, torch.tensor(row.label_group)

df_sub = pd.read_csv('./train.csv')

df_train = df_sub.copy()
df_train['filepath'] = df_train['image'].apply(lambda x: os.path.join('./', 'train_images', x))

dataset_train = LandmarkDataset(df_train, 'train', 'train')
test_loader = DataLoader(dataset_train, batch_size=128, num_workers=8, shuffle=True,pin_memory=True)


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-8, betas=(0.9, 0.98), eps=1e-6,
                       weight_decay=0.001)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

for epoch in range(EPOCH):
    for i, batch in enumerate(test_loader):
        optimizer.zero_grad()

        data = batch
        data_images = data["P"].to(device)
        data_texts = data["T"]
        images = data_images
        if len(data_texts) >77:
            data_texts = data_texts[:76]
        texts = clip.tokenize(data_texts).to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss.backward()

        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
        print(f"[{epoch}]-[{i}]: {total_loss.item()}")

torch.save(model, './models/model_clip.pkl')
# torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': total_loss,
#         }, f"models/model_fscoco.pt") #just change to your preferred folder/filename
print(f"{EPOCH} model have saved")

# load data

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path = [
    './geffnet-20200820'
] + sys.path
#配置环境
import numpy as np, pandas as pd, gc
import cv2, matplotlib.pyplot as plt
import cudf, cuml, cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors

import albumentations
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

import geffnet
from transformers import *
import pickle
import cupy as cp
COMPUTE_CV = True

test = pd.read_csv('./test.csv')
if len(test)>3: COMPUTE_CV = False
else: print('this submission notebook will compute CV score, but commit notebook will not')

train = pd.read_csv('./train.csv')
tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
train['target'] = train.label_group.map(tmp)
print('train shape is', train.shape )
train.head()

tmp = train.groupby('image_phash').posting_id.agg('unique').to_dict()
train['oof'] = train.image_phash.map(tmp)

def getMetric(col):
    def f1score(row):
        n = len( np.intersect1d(row.target,row[col]) )
        return 2*n / (len(row.target)+len(row[col]))
    return f1score

train['f1'] = train.apply(getMetric('oof'),axis=1)
print('CV score for baseline =',train.f1.mean())

if COMPUTE_CV:
    test = pd.read_csv('./train_fold.csv')
#     test = test[test.fold==0]
    test_gf = cudf.DataFrame(test)
    print('Using train as test to compute CV (since commit notebook). Shape is', test_gf.shape )
else:
    test = pd.read_csv('./test.csv')
    test_gf = cudf.read_csv('./test.csv')
    print('Test shape is', test_gf.shape )
test_gf.head()

import os


def get_transforms(img_size=256):
    return albumentations.Compose([
        albumentations.Resize(img_size, img_size),
        albumentations.Normalize()
    ])


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

tokenizer = AutoTokenizer.from_pretrained('./bert base uncased')

if not COMPUTE_CV:
    df_sub = pd.read_csv('./test.csv')

    df_test = df_sub.copy()
    df_test['filepath'] = df_test['image'].apply(lambda x: os.path.join('./', 'test_images', x))

    dataset_test = LandmarkDataset(df_test, 'test', 'test', transforms=get_transforms(img_size=256), tokenizer=tokenizer)
    test_loader = DataLoader(dataset_test, batch_size=16, num_workers=4)

    print(len(dataset_test),dataset_test[0])
else:
    df_sub = test

    df_test = df_sub.copy()
    df_test['filepath'] = df_test['image'].apply(lambda x: os.path.join('./', 'train_images', x))

    dataset_test = LandmarkDataset(df_test, 'test', 'test', transforms=get_transforms(img_size=256), tokenizer=tokenizer)
    test_loader = DataLoader(dataset_test, batch_size=16, num_workers=4)

    print(len(dataset_test),dataset_test[0])


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


sigmoid = torch.nn.Sigmoid()


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class enet_arcface_FINAL(nn.Module):

    def __init__(self, enet_type, out_dim):
        super(enet_arcface_FINAL, self).__init__()
        self.bert = AutoModel.from_pretrained('./bert base uncased')
        self.enet = geffnet.create_model(enet_type.replace('-', '_'), pretrained=None)
        self.feat = nn.Linear(self.enet.classifier.in_features + self.bert.config.hidden_size, 512)
        self.swish = Swish_module()
        self.dropout = nn.Dropout(0.5)
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.enet.classifier = nn.Identity()

    def forward(self, x, input_ids, attention_mask):
        x = self.enet(x)
        text = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        x = torch.cat([x, text], 1)
        x = self.swish(self.feat(x))
        return F.normalize(x), self.metric_classify(x)


def load_model(model, model_file):
    state_dict = torch.load(model_file)
    if "model_state_dict" in state_dict.keys():
        state_dict = state_dict["model_state_dict"]
    state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
    #     del state_dict['metric_classify.weight']
    model.load_state_dict(state_dict, strict=False)
    print(f"loaded {model_file}")
    model.eval()
    return model


import math
from tqdm import tqdm

# WGT = './b0ns_256_bert_20ep_fold0_epoch27.pth'
#
# model = enet_arcface_FINAL('tf_efficientnet_b0_ns', out_dim=11014).cuda()
# model = load_model(model, WGT)
#
# embeds = []
#
# with torch.no_grad():
#     for img, input_ids, attention_mask in tqdm(test_loader):
#         img, input_ids, attention_mask = img.cuda(), input_ids.cuda(), attention_mask.cuda()
#         feat, _ = model(img, input_ids, attention_mask)
#         image_embeddings = feat.detach().cpu().numpy()
#         embeds.append(image_embeddings)
#
# del model
# _ = gc.collect()
# image_embeddings = np.concatenate(embeds)


# with open('image_embeddings.pkl', 'wb') as f:    #Pickling
#     pickle.dump(image_embeddings, f)

# with open('image_embeddings.pkl', 'rb') as f:    # Unpickling
#     image_embeddings = pickle.load(f)
#
# KNN = 50
# if len(test)==3: KNN = 2
# model = NearestNeighbors(n_neighbors=KNN)
# model.fit(image_embeddings)
#
#
# import pandas as pd
# import cupy as cp
#
#
# # 假设 image_embeddings 是图像的嵌入向量
# image_embeddings = cp.array(image_embeddings)  # 使用了 CuPy 库来进行大规模向量化计算
# threshold = 0.475
#
# print(f"threshold: {threshold}")
# preds = []
# CHUNK = 1024 * 4
# print('Finding similar images...')
# CTS = len(image_embeddings) // CHUNK
# if len(image_embeddings) % CHUNK != 0:
#     CTS += 1
#
# for j in range(CTS):
#     a = j * CHUNK
#     b = min((j + 1) * CHUNK, len(image_embeddings))
#     print('chunk', a, 'to', b)
#     # 寻找相似的邻居
#     distances, indices = model.kneighbors(image_embeddings[a:b], n_neighbors=KNN)
#     # 将距离转换为相似度
#     similarities = 1 / (1 + distances)
#
#     for k in range(b - a):
#         IDX = cp.where(similarities[k,] > threshold)[0]
#         o = test.iloc[cp.asnumpy(indices[k, IDX])].posting_id.values
#         preds.append(o)
#
#     del distances, indices
# test['preds2'] = preds
# test.head()
#
# image_embeddings = image_embeddings.get()
# del image_embeddings
# cp.get_default_memory_pool().free_all_blocks()  # 释放显存
# _ = gc.collect()

print('Computing text embeddings...')


class CFG:
    # data augmentation
    IMG_SIZE = 512
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    SEED = 2023

    # data split
    N_SPLITS = 5
    TEST_FOLD = 0
    VALID_FOLD = 1

    EPOCHS = 8
    BATCH_SIZE = 8

    NUM_WORKERS = 4
    DEVICE = "cuda:0"

    CLASSES = 11014  # !!!!!!!!!!!!!
    SCALE = 30
    MARGIN = 0.6

    SCHEDULER_PARAMS = {
        "lr_start": 1e-5,
        "lr_max": 1e-5 * 32,
        "lr_min": 1e-6,
        "lr_ramp_ep": 5,
        "lr_sus_ep": 0,
        "lr_decay": 0.8,
    }

    MODEL_NAME = "eca_nfnet_l0"
    FC_DIM = 512
    MODEL_PATH = './bert base uncased'
    FEAT_PATH = f"../input/shopee-embeddings/{MODEL_NAME}_arcface.npy"
### Dataset

class TitleDataset(Dataset):
    def __init__(self, df, text_column, label_column):
        texts = df[text_column]
        self.labels = df[label_column].values

        self.titles = []
        for title in texts:
            title = title.encode('latin1').decode('unicode-escape').encode('latin1').decode('utf-8')
            title = title.lower()
            self.titles.append(title)
    def __len__(self):
        return len(self.titles)
    def __getitem__(self, idx):
        text = self.titles[idx]
        label = torch.tensor(self.labels[idx])
        return text, label

title_dataset = TitleDataset(test, 'title', 'label_group', tokenizer)
title_loader = DataLoader(title_dataset, batch_size=16, num_workers=4)


model_name = './bert base uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).cuda()
# 准备输入数据

embeddings = []
model.eval()
with torch.no_grad():
    for text, _ in tqdm(title_loader):
        text.cuda()
        tokens = tokenizer(text, padding='max_length', truncation=True, max_length=16, return_tensors="pt")
        outputs = model(**tokens)
        embeddings.append(outputs.last_hidden_state.detach().cpu().numpy())


del model
_ = gc.collect()
text_embeddings = np.concatenate(embeddings,axis=0)
print(text_embeddings.shape)

# model = TfidfVectorizer(stop_words=None,
#                         binary=True,
#                         max_features=25000)
# text_embeddings = model.fit_transform(test_gf.title).toarray()
# print('text embeddings shape', text_embeddings.shape)
#
#
# with open('text_embeddings.pkl', 'wb') as f:    #Pickling
#     pickle.dump(text_embeddings, f)
#
# with open('text_embeddings.pkl', 'rb') as f:    # Unpickling
#     text_embeddings = pickle.load(f)

KNN = 50
if len(test)==3: KNN = 2
model = NearestNeighbors(n_neighbors=KNN)
model.fit(text_embeddings)

text_embeddings = cp.array(text_embeddings)  # 使用了 CuPy 库来进行大规模向量化计算

tmp = test.groupby('label_group').posting_id.agg('unique').to_dict()
test['target'] = test.label_group.map(tmp)
threshold = 0.508
print(f"threshold: {threshold}")
preds = []
CHUNK = 1024 * 4

print('Finding similar titles...')
CTS = len(test) // CHUNK
if len(test) % CHUNK != 0: CTS += 1
for j in range(CTS):
    a = j * CHUNK
    b = min((j + 1) * CHUNK, len(test))
    print('chunk', a, 'to', b)
    # 寻找相似的邻居
    distances, indices = model.kneighbors(text_embeddings[a:b], n_neighbors=KNN)
    # 将距离转换为相似度
    similarities = 1 / (1 + distances)

    for k in range(b - a):
        IDX = cp.where(similarities[k,] > threshold)[0]
        o = test.iloc[cp.asnumpy(indices[k, IDX])].posting_id.values
        preds.append(o)

    del distances, indices

text_embeddings = text_embeddings.get()
del text_embeddings
cp.get_default_memory_pool().free_all_blocks()  # 释放显存
_ = gc.collect()

test['preds'] = preds
test.head()


# tmp = test.groupby('image_phash').posting_id.agg('unique').to_dict()
# test['preds3'] = test.image_phash.map(tmp)
# test.head()


def combine_for_sub(row):
    x = np.concatenate([row.preds, row.preds2])
    return ' '.join(np.unique(x))


def combine_for_cv(row):
    x = np.concatenate([row.preds, row.preds2])
    return np.unique(x)


if COMPUTE_CV:
    tmp = test.groupby('label_group').posting_id.agg('unique').to_dict()
    test['target'] = test.label_group.map(tmp)
    test['oof'] = test.apply(combine_for_cv, axis=1)
    test['f1'] = test.apply(getMetric('oof'), axis=1)
    print('CV Score =', test.f1.mean())

test['matches'] = test.apply(combine_for_sub, axis=1)

print("CV for image :", round(test.apply(getMetric('preds2'), axis=1).mean(), 3))
print("CV for text  :", round(test.apply(getMetric('preds'), axis=1).mean(), 3))
# print("CV for phash :", round(test.apply(getMetric('preds3'), axis=1).mean(), 3))

test

test[['posting_id', 'matches']].to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv')
sub.head()

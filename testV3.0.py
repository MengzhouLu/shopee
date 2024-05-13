import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path = [
    './geffnet-20200820'
] + sys.path
#配置环境
import regex
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

WGT = './b0ns_256_bert_20ep_fold0_epoch27.pth'

model = enet_arcface_FINAL('tf_efficientnet_b0_ns', out_dim=11014).cuda()
image_model = load_model(model, WGT)
#
# embeds = []
#
# with torch.no_grad():
#     for img, input_ids, attention_mask in tqdm(test_loader):
#         img, input_ids, attention_mask = img.cuda(), input_ids.cuda(), attention_mask.cuda()
#         feat, _ = image_model(img, input_ids, attention_mask)
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
# img_embs=image_embeddings
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



import re

def extract_and_replace_with_standard_unit(tit):
    tit = ' ' + tit.lower() + ' '
    for cat, units in measurements.items():
        for unit_name, mult in units:
            pat = fr'\b(\d+(?:[\,\.]\d+)?) ?{unit_name}s?\b'
            matches = re.finditer(pat, tit)
            for match in matches:
                num = to_num(match.group(1), mult)
                tit = re.sub(fr'\b{match.group(1)} ?{unit_name}s?\b', f"{num} {cat}", tit)
    return tit.strip()




### Dataset

class TitleDataset(Dataset):
    def __init__(self, df, text_column, label_column):
        texts = df[text_column]
        self.labels = df[label_column].values

        self.titles = []
        for title in texts:
            title = title.encode('latin1').decode('unicode-escape').encode('latin1').decode('utf-8')
            title = title.lower()
            title=extract_and_replace_with_standard_unit(title)
            self.titles.append(title)
    def __len__(self):
        return len(self.titles)
    def __getitem__(self, idx):
        text = self.titles[idx]
        label = torch.tensor(self.labels[idx])
        return text, label

title_dataset = TitleDataset(test, 'title', 'label_group')
title_loader = DataLoader(title_dataset, batch_size=128, num_workers=4)

print(title_dataset[:20])



model_name = './bertmodel'
tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
state = torch.load('./bertmodel/bert_indo_val0.pth')
model.load_state_dict(state,strict=False)
model.cuda()
bert_model=model
# 准备输入数据

embeddings = []
model.eval()
with torch.no_grad():
    for title, _ in tqdm(title_loader):
        tokens = tokenizer(title, padding='max_length', truncation=True, max_length=100, return_tensors="pt").to('cuda')
        outputs = bert_model(**tokens)
        sentence_embeddings = outputs.last_hidden_state[:, 0, :]  # 获取[CLS]标记所对应的输出
        # embeddings.append(sentence_embeddings.cpu().numpy())
        embeddings.append(sentence_embeddings.detach().cpu())


embeddings=F.normalize(torch.cat(embeddings, dim=0),dim=1).numpy()
print(embeddings.shape)
with open('title_embeddings.pkl', 'wb') as f:    #Pickling
    pickle.dump(embeddings, f)

with open('title_embeddings.pkl', 'rb') as f:    # Unpickling
    text_embeddings = pickle.load(f)

del model
_ = gc.collect()
# text_embeddings = np.concatenate(embeddings, axis=0)
# text_embeddings = text_embeddings.reshape(test.shape[0], -1)
print(text_embeddings.shape)

bert_embs=text_embeddings
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
threshold=0.58
print(f"threshold: {threshold}")
preds = []
CHUNK = 1024 * 4*4

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




tmp = test.groupby('image_phash').posting_id.agg('unique').to_dict()
test['preds3'] = test.image_phash.map(tmp)
test.head()


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
print("CV for phash :", round(test.apply(getMetric('preds3'), axis=1).mean(), 3))

test

test[['posting_id', 'matches']].to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv')
sub.head()

def add_target_groups(data_df, source_column='label_group', target_column='target'):
    target_groups = data_df.groupby(source_column).indices
    data_df[target_column]=data_df[source_column].map(target_groups)
    return data_df

def add_splits(train_df, valid_group=0):
    grouped = train_df.groupby('label_group').size()

    labels, sizes =grouped.index.to_list(), grouped.to_list()

    skf = StratifiedKFold(5)
    splits = list(skf.split(labels, sizes))

    group_to_split =  dict()
    for idx in range(5):
        labs = np.array(labels)[splits[idx][1]]
        group_to_split.update(dict(zip(labs, [idx]*len(labs))))

    train_df['split'] = train_df.label_group.replace(group_to_split)
    train_df['is_valid'] = train_df['split'] == valid_group
    return train_df

def embs_from_model(model, dl):
    all_embs = []
    all_ys=[]
    for batch in tqdm(dl):
        if len(batch) ==2:
            bx,by=batch
        else:
            bx,=batch
            by=torch.zeros(1)
        with torch.no_grad():
            embs = model(bx)
            all_embs.append(embs.half())
        all_ys.append(by)
    all_embs = F.normalize(torch.cat(all_embs))
    return all_embs, torch.cat(all_ys)

def get_targets_shape(train_df):
    all_targets = add_target_groups(train_df).target.to_list()
    all_targets_lens = [len(t) for t in all_targets]
    targets_shape = []
    for size in range(min(all_targets_lens), max(all_targets_lens)+1):
        count = all_targets_lens.count(size) / len(all_targets)
        targets_shape.append((size,count))
    return targets_shape

def chisel(groups, groups_p, pos, target_count):
    probs = []
    groups_lens = [len(g)for g in groups]
    current_count = groups_lens.count(pos)
    if current_count >= target_count:

        return
    to_cut = target_count - current_count
    for i in range(len(groups)):
        if len(groups_p[i])>pos:
            probs.append((i, groups_p[i][pos]))
    probs.sort(key=lambda x:x[1])
    for i in range(min(to_cut, len(probs))):
        group_idx = probs[i][0]
        groups[group_idx]=groups[group_idx][:pos]
        groups_p[group_idx]=groups_p[group_idx][:pos]

def sorted_pairs(distances, indices):
    triplets = []
    n= len(distances)
    for x in range(n):
        used=set()
        for ind, dist in zip(indices[x].tolist(), distances[x].tolist()):
            if not ind in used:
                triplets.append((x, ind, dist))
                used.add(ind)
    return sorted(triplets, key=lambda x: -x[2])
def do_chunk(embs):
    step = 1000
    for chunk_start in range(0, embs.shape[0], step):
        chunk_end = min(chunk_start+step, len(embs))
        yield embs[chunk_start:chunk_end]
def get_nearest(embs, emb_chunks, K=None, sorted=True):
    if K is None:
        K = min(51, len(embs))
    distances = []
    indices = []
    for chunk in emb_chunks:
        sim = embs @ chunk.T
        top_vals, top_inds = sim.topk(K, dim=0, sorted=sorted)
        distances.append(top_vals.T)
        indices.append(top_inds.T)
    return torch.cat(distances), torch.cat(indices)

def combined_distances(embs_list):
    K = min(len(embs_list[0]), 51)
    combined_inds =[get_nearest(embs, do_chunk(embs))[1] for embs in embs_list]
    combined_inds = torch.cat(combined_inds, dim=1)
    res_inds,res_dists = [],[]
    for x in range(len(combined_inds)):
        inds = combined_inds[x].unique()
        Ds = [embs[None,x] @ embs[inds].T for embs in embs_list]
        D = Ds[0] + Ds[1] - Ds[0] * Ds[1]
        top_dists, top_inds = D.topk(K)
        res_inds.append(inds[top_inds])
        res_dists.append(top_dists)
    return torch.cat(res_inds), torch.cat(res_dists)

def blend_embs(embs_list, threshold, m2_threshold, data_df):
    combined_inds, combined_dists = combined_distances(embs_list)
    check_measurements(combined_dists, combined_inds, data_df)
    new_embs_list = L((torch.empty_like(embs) for embs in embs_list))
    for x in range(len(embs_list[0])):
        neighs = combined_dists[x] > threshold
        if neighs.sum() == 1 and combined_dists[x][1]>m2_threshold:
            neighs[1]=1
        neigh_inds, neigh_ratios = combined_inds[x, neighs], combined_dists[x,neighs]
        for embs, new_embs in zip(embs_list, new_embs_list):
            new_embs[x] = (embs[neigh_inds] * neigh_ratios.view(-1,1)).sum(dim=0)
    return new_embs_list.map(F.normalize)

# Not used in solution, just for illustration purpose
def f1(tp, fp, num_tar):
    return 2 * tp / (tp+fp+num_tar)

def build_from_pairs(pairs, target, display = True):
    score =0
    tp = [0]*len(target)
    fp = [0]*len(target)
    scores=[]
    vs=[]
    group_sizes = [len(x) for x in target]
    for x, y, v in pairs:
        group_size = group_sizes[x]
        score -= f1(tp[x], fp[x], group_size)
        if y in target[x]: tp[x] +=1
        else: fp[x] +=1
        score += f1(tp[x], fp[x], group_size)
        scores.append(score / len(target))
        vs.append(v)
    if display:
        plt.plot(scores)
        am =torch.tensor(scores).argmax()
        print(f'{scores[am]:.3f} at {am/len(target)} pairs or {vs[am]:.3f} threshold')
    return scores


def score_distances(dist, targets, display=False):
    triplets = dist_to_edges(dist)[:len(dist)*10]
    return max(build_from_pairs(triplets, targets, display))

def score_group(group, target):
    tp = len(set(group).intersection(set(target)))
    return 2 * tp / (len(group)+len(target))
def score_all_groups(groups, targets):
    scores = [score_group(groups[i], targets[i]) for i in range(len(groups))]
    return sum(scores)/len(scores)
def show_groups(groups, targets):
    groups_lens = [len(g)for g in groups]
    targets_lens = [len(g) for g in targets]
    plt.figure(figsize=(8,8))
    plt.hist((groups_lens,targets_lens) ,bins=list(range(1,52)), label=['preds', 'targets'])
    plt.legend()
    plt.title(f'score: {score_all_groups(groups, targets):.3f}')
    plt.show()



def add_measurements(data):
    data['measurement'] = data.title.map(extract)
    return data

def match_measures(m1, m2):
    k1,k2 = set(m1.keys()), set(m2.keys())
    common = k1.intersection(k2)
    if not common: return True
    for key in common:
        s1,s2 = m1[key], m2[key]
        if s1.intersection(s2):
            return True
    return False

def check_measurements(combined_dists, combined_inds, data_df):
    K = min(8, len(data_df)) * len(data_df)
    _, inds_k = combined_dists.view(-1).topk(K)
    removed = 0
    inds_k = inds_k.tolist()
    for idx in inds_k:
        x = idx // combined_inds.shape[1]
        y_idx = idx % combined_inds.shape[1]
        y = combined_inds[x,y_idx]
        if not match_measures(data_df.iloc[x].measurement, data_df.iloc[y.item()].measurement):
            removed +=1
            combined_dists[x][y_idx]=0
    print('removed', removed, 'matches')

img_embs,ys = embs_from_model(image_model, test_loader)
bert_embs, ys = embs_from_model(bert_model, title_loader.valid)


valid_df=test
# thees are thresholds used for Neighborhood Blending
RECIPROCAL_THRESHOLD = .97
MIN_PAIR_THRESHOLD = .6

new_embs = blend_embs([img_embs, bert_embs], RECIPROCAL_THRESHOLD, MIN_PAIR_THRESHOLD, valid_df)

set_size = len(img_embs)
target_matrix= ys[:,None]==ys[None,:]
targets = [torch.where(t)[0].tolist() for t in target_matrix]

combined_inds, combined_dists = combined_distances([img_embs, bert_embs])
pairs = sorted_pairs(combined_dists, combined_inds)
_=build_from_pairs(pairs[:10*len(combined_inds)], targets)

check_measurements(combined_dists, combined_inds, valid_df)
pairs = sorted_pairs(combined_dists, combined_inds)
_=build_from_pairs(pairs[:10*len(combined_inds)], targets)
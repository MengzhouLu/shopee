import os

import cv2
import pandas as pd
from tqdm import tqdm
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

# # load clip model
# device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
# model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training

# 加载已经训练好的模型
# load model
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training
checkpoint = torch.load("./model_clip.pkl")
model.load_state_dict(checkpoint.state_dict())

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
        text= text.encode('latin1').decode('unicode-escape').encode('latin1').decode('utf-8')
        text = text.lower()
        text = extract_and_replace_with_standard_unit(text)
        if self.mode == 'train':
            if len(text) > 77:#train时，截断到77个字
                text = text[:77]
        else:
            if len(text) > 64:#test时，截断到16个字
                text = text[:64]
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
train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=16, shuffle=True,pin_memory=True)

dataset_test = LandmarkDataset(df_train, 'train', 'test')#这里的mode是test，所以会截断到63个字
test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=16, shuffle=True,pin_memory=True)

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


#测试CLIP能不能找到最能描述图片的文字：正确率80%左右
def test_model(model, test_loader):
    model.eval()
    count_miss = 0
    for batch in tqdm(test_loader):

        data = batch
        data_images = data["P"].to(device)
        data_texts = data["T"]
        texts = [f"a photo of a {title}" for title in data_texts]
        texts_tokenized = clip.tokenize(texts).to(device)
        with torch.no_grad():
            image_features = model.encode_image(data_images).float()
            text_features = model.encode_text(texts_tokenized).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)#矩阵乘法相当于计算图文的相似度 softmax将相似度转化为概率
            top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
        for i in range(len(data_texts)):
            # print(f"Image Text: {data_texts[i]}")
            # print("Predicted Texts:")
            label=[data_texts[label_idx] for label_idx in top_labels[i]]
            # for label_idx in top_labels[i]:
            #     print(data_texts[label_idx])
            if data_texts[i] not in label:
                count_miss += 1

            # print("---------")

    print(f"Total miss count: {count_miss}")
    accuracy = 1 - count_miss / len(test_loader.dataset)
    print(f"Accuracy: {accuracy * 100:.2f}%")


## train model
if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-8, betas=(0.9, 0.98), eps=1e-6,
                       weight_decay=0.001)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

# for epoch in range(EPOCH):
#     loss=[]
#     for batch in tqdm(train_loader):
#         optimizer.zero_grad()
#
#         data = batch
#         data_images = data["P"].to(device)
#         data_texts = data["T"]
#         images = data_images
#         texts = clip.tokenize(data_texts).to(device)
#
#         logits_per_image, logits_per_text = model(images, texts)
#
#         ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
#
#         total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
#         loss.append(total_loss.item())
#         total_loss.backward()
#
#         if device == "cpu":
#             optimizer.step()
#         else:
#             convert_models_to_fp32(model)
#             optimizer.step()
#             clip.model.convert_weights(model)
#         # print(f"[{epoch}]-[{i}]: {total_loss.item()}")
#     print(f"[{epoch}]-[mean loss]: {np.mean(loss)}")
#     if epoch % 5 == 0:
#         test_model(model,test_loader)
#
# torch.save(model, './model_clip.pkl')
# # torch.save({
# #         'epoch': epoch,
# #         'model_state_dict': model.state_dict(),
# #         'optimizer_state_dict': optimizer.state_dict(),
# #         'loss': total_loss,
# #         }, f"models/model_fscoco.pt") #just change to your preferred folder/filename
# print(f"{EPOCH} model have saved")


img_embeddings=[]
text_embeddings=[]
combine_embeddings=[]
#由CLIP模型得到的图片和文本的嵌入向量
def embs_from_model(model, test_loader):
    model.eval()

    for batch in tqdm(test_loader):

        data = batch
        data_images = data["P"].to(device)
        data_texts = data["T"]
        # texts = [f"a photo of a {title}" for title in data_texts]
        texts = data_texts
        texts_tokenized = clip.tokenize(texts).to(device)
        with torch.no_grad():
            image_features = model.encode_image(data_images).float()
            text_features = model.encode_text(texts_tokenized).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            img_embeddings.append(image_features.cpu().numpy())
            text_embeddings.append(text_features.cpu().numpy())
            combined_features = torch.cat((image_features, text_features), dim=1)
            combine_embeddings.append(combined_features.cpu().numpy())  # 拼接两个向量，作为一个新的向量
            print(image_features.shape,text_features.shape,combined_features.shape)

embs_from_model(model, test_loader)

img_embeddings=np.concatenate(img_embeddings,axis=0)
text_embeddings=np.concatenate(text_embeddings,axis=0)
combine_embeddings=np.concatenate(combine_embeddings,axis=0)
print(img_embeddings.shape,text_embeddings.shape,combine_embeddings.shape)

import pickle
with open('image_embeddings_clip.pkl', 'wb') as f:    #Pickling
    pickle.dump(img_embeddings, f)
with open('text_embeddings_clip.pkl', 'wb') as f:    #Pickling
    pickle.dump(text_embeddings, f)
# with open('combine_embeddings_clip.pkl', 'wb') as f:    #Pickling
#     pickle.dump(combine_embeddings, f)

# with open('image_embeddings_clip.pkl', 'rb') as f:    # Unpickling
#     image_embeddings = pickle.load(f)
# with open('text_embeddings_clip.pkl', 'rb') as f:    # Unpickling
#     text_embeddings = pickle.load(f)
# with open('combine_embeddings_clip.pkl', 'rb') as f:    # Unpickling
#     combine_embeddings = pickle.load(f)
#
# def getMetric(col):
#     def f1score(row):
#         n = len( np.intersect1d(row.target,row[col]) )
#         return 2*n / (len(row.target)+len(row[col]))
#     return f1score
#
#
#
# import cudf
# test = pd.read_csv('./train_fold.csv')
# test_gf = cudf.DataFrame(test)
# print('Using train as test to compute CV (since commit notebook). Shape is', test_gf.shape )
# print('Test shape is', test_gf.shape)
#
#
#
#
#
# def combine_for_sub(row):
#     x = np.concatenate([row.preds, row.preds2])
#     return ' '.join(np.unique(x))
#
#
# def combine_for_cv(row):
#     x = np.concatenate([row.preds, row.preds2])
#     return np.unique(x)
#
#
# if True:
#     tmp = test.groupby('label_group').posting_id.agg('unique').to_dict()
#     test['target'] = test.label_group.map(tmp)
#
#
#
#
# from cuml.neighbors import NearestNeighbors
#
# KNN = 50
# if len(test)==3: KNN = 2
# model = NearestNeighbors(n_neighbors=KNN)
# model.fit(image_embeddings)
#
#
#
# import cupy as cp
# import numpy as np, pandas as pd, gc
#
# # 假设 image_embeddings 是图像的嵌入向量
# image_embeddings = cp.array(image_embeddings)  # 使用了 CuPy 库来进行大规模向量化计算
#
# for threshold in [0.99,0.992,0.994,0.996,0.998,0.999]:
#
#     print(f"threshold: {threshold}")
#     preds = []
#     CHUNK = 1024 * 4
#     print('Finding similar images...')
#     CTS = len(image_embeddings) // CHUNK
#     if len(image_embeddings) % CHUNK != 0:
#         CTS += 1
#
#     for j in range(CTS):
#         a = j * CHUNK
#         b = min((j + 1) * CHUNK, len(image_embeddings))
#         print('chunk', a, 'to', b)
#         # 寻找相似的邻居
#         distances, indices = model.kneighbors(image_embeddings[a:b], n_neighbors=KNN)
#         # 将距离转换为相似度
#         similarities = 1 / (1 + distances)
#
#         for k in range(b - a):
#             IDX = cp.where(similarities[k,] > threshold)[0]
#             o = test.iloc[cp.asnumpy(indices[k, IDX])].posting_id.values
#             preds.append(o)
#
#         del distances, indices
#     test['preds2'] = preds
#     test.head()
#     print("CV for image :", round(test.apply(getMetric('preds2'), axis=1).mean(), 3))
#
# image_embeddings = image_embeddings.get()
# del image_embeddings
# cp.get_default_memory_pool().free_all_blocks()  # 释放显存
# _ = gc.collect()
#
#
#
#
# KNN = 50
# if len(test)==3: KNN = 2
# model = NearestNeighbors(n_neighbors=KNN)
# model.fit(text_embeddings)
#
# text_embeddings = cp.array(text_embeddings)  # 使用了 CuPy 库来进行大规模向量化计算
#
# tmp = test.groupby('label_group').posting_id.agg('unique').to_dict()
# test['target'] = test.label_group.map(tmp)
# print('----------------------------------------')
# for threshold in [0.99,0.992,0.994,0.996,0.998,0.999]:
#     print(f"threshold: {threshold}")
#     preds = []
#     CHUNK = 1024 * 4 * 4
#
#     print('Finding similar titles...')
#     CTS = len(test) // CHUNK
#     if len(test) % CHUNK != 0: CTS += 1
#     for j in range(CTS):
#         a = j * CHUNK
#         b = min((j + 1) * CHUNK, len(test))
#         print('chunk', a, 'to', b)
#         # 寻找相似的邻居
#         distances, indices = model.kneighbors(text_embeddings[a:b], n_neighbors=KNN)
#         # 将距离转换为相似度
#         similarities = 1 / (1 + distances)
#
#         for k in range(b - a):
#             IDX = cp.where(similarities[k,] > threshold)[0]
#             o = test.iloc[cp.asnumpy(indices[k, IDX])].posting_id.values
#             preds.append(o)
#
#         del distances, indices
#     test['preds'] = preds
#     test.head()
#     print("CV for text  :", round(test.apply(getMetric('preds'), axis=1).mean(), 3))
#
# text_embeddings = text_embeddings.get()
# del text_embeddings
# cp.get_default_memory_pool().free_all_blocks()  # 释放显存
# _ = gc.collect()
#
#
#
#
# KNN = 50
# if len(test)==3: KNN = 2
# model = NearestNeighbors(n_neighbors=KNN)
# model.fit(combine_embeddings)
#
# combine_embeddings = cp.array(combine_embeddings)  # 使用了 CuPy 库来进行大规模向量化计算
#
# tmp = test.groupby('label_group').posting_id.agg('unique').to_dict()
# test['target'] = test.label_group.map(tmp)
#
#
# print('----------------------------------------')
# for threshold in [0.99,0.992,0.994,0.996,0.998,0.999]:
#     print(f"threshold: {threshold}")
#     preds = []
#     CHUNK = 1024 * 4 * 4
#
#     print('Finding combine...')
#     CTS = len(test) // CHUNK
#     if len(test) % CHUNK != 0: CTS += 1
#     for j in range(CTS):
#         a = j * CHUNK
#         b = min((j + 1) * CHUNK, len(test))
#         print('chunk', a, 'to', b)
#         # 寻找相似的邻居
#         distances, indices = model.kneighbors(combine_embeddings[a:b], n_neighbors=KNN)
#         # 将距离转换为相似度
#         similarities = 1 / (1 + distances)
#
#         for k in range(b - a):
#             IDX = cp.where(similarities[k,] > threshold)[0]
#             o = test.iloc[cp.asnumpy(indices[k, IDX])].posting_id.values
#             preds.append(o)
#
#         del distances, indices
#     test['preds3'] = preds
#     test.head()
#     print("CV for combine:", round(test.apply(getMetric('preds3'), axis=1).mean(), 3))
#
#
#
# combine_embeddings = combine_embeddings.get()
# del combine_embeddings
# cp.get_default_memory_pool().free_all_blocks()  # 释放显存
# _ = gc.collect()
#
#
#
#
#
#
#
#
#
# def combine_for_sub(row):
#     x = np.concatenate([row.preds, row.preds2])
#     return ' '.join(np.unique(x))
#
#
# def combine_for_cv(row):
#     x = np.concatenate([row.preds, row.preds2])
#     return np.unique(x)
#
#
# if True:
#     tmp = test.groupby('label_group').posting_id.agg('unique').to_dict()
#     test['target'] = test.label_group.map(tmp)
#     test['oof'] = test.apply(combine_for_cv, axis=1)
#     test['f1'] = test.apply(getMetric('oof'), axis=1)
#     print('CV Score =', test.f1.mean())
#
# test['matches'] = test.apply(combine_for_sub, axis=1)
#
# print("CV for image :", round(test.apply(getMetric('preds2'), axis=1).mean(), 3))
# print("CV for text  :", round(test.apply(getMetric('preds'), axis=1).mean(), 3))
# print("CV for combine:", round(test.apply(getMetric('preds3'), axis=1).mean(), 3))
#
# test[['posting_id','target','preds2','preds','preds3']].to_csv('submission_clip.csv',index=False)
# pd.read_csv('submission_clip.csv').head()

test = pd.read_csv('./train.csv')
def test2_model():
    with open('image_embeddings_clip.pkl', 'rb') as f:  # Unpickling
        image_embeddings = pickle.load(f)
    with open('text_embeddings_clip.pkl', 'rb') as f:  # Unpickling
        text_embeddings = pickle.load(f)
    with open('combine_embeddings_clip.pkl', 'rb') as f:  # Unpickling
        combine_embeddings = pickle.load(f)#图文拼接
    with torch.no_grad():
        image_embeddings = torch.from_numpy(image_embeddings).to(device)
        # image_embeddings=image_embeddings.half()#调整精度
        image_probs = (100.0 * image_embeddings @ image_embeddings.T)
        image_prob = image_probs.detach().cpu()
        del image_probs
        torch.cuda.empty_cache() # 释放显存
        top_probs, top_labels = image_prob.softmax(dim=-1).topk(5, dim=-1)
        print(image_prob.shape, top_probs.shape,top_labels.shape)  # torch.Size([34250, 34250]) torch.Size([34250, 5]) torch.Size([34250, 5])
        print('图像相似度')
        for i in range(10):
            print(top_probs[i])
        print('--------------------------------------------')

        text_embeddings = torch.from_numpy(text_embeddings).to(device)
        text_probs = (100.0 * text_embeddings @ text_embeddings.T)
        text_prob = text_probs.detach().cpu()
        del text_probs
        torch.cuda.empty_cache() # 释放显存
        top_probs, top_labels = text_prob.softmax(dim=-1).topk(5, dim=-1)
        print(text_prob.shape, top_probs.shape,top_labels.shape)  # torch.Size([34250, 34250]) torch.Size([34250, 5]) torch.Size([34250, 5])
        print('文本相似度')
        for i in range(10):
            print(top_probs[i])
            print(test.iloc[top_labels[i]]['title'].values)
        print('--------------------------------------------')

        combine_embeddings = torch.from_numpy(combine_embeddings).to(device)
        combine_probs = (100.0 * combine_embeddings @ combine_embeddings.T)
        combine_prob = combine_probs.detach().cpu()
        del combine_probs
        torch.cuda.empty_cache() # 释放显存
        top_probs, top_labels = combine_prob.softmax(dim=-1).topk(5, dim=-1)
        print(combine_prob.shape, top_probs.shape,top_labels.shape)  # torch.Size([34250, 34250]) torch.Size([34250, 5]) torch.Size([34250, 5])
        print('图文相似度(图文拼接)')
        for i in range(10):
            print(top_probs[i])
        print('--------------------------------------------')

        fix_prob=image_prob+text_prob
        top_probs, top_labels = fix_prob.softmax(dim=-1).topk(5, dim=-1)
        print(fix_prob.shape, top_probs.shape,top_labels.shape)  # torch.Size([34250, 34250]) torch.Size([34250, 5]) torch.Size([34250, 5])
        print('图文相似度(图文相加)')
        for i in range(10):
            print(top_probs[i])
        input()

    # model.eval()
    # count_miss = 0
    # for batch in tqdm(test_loader):
    #
    #     data = batch
    #     data_images = data["P"].to(device)
    #     data_texts = data["T"]
    #     texts = [f"a photo of a {title}" for title in data_texts]
    #     texts_tokenized = clip.tokenize(texts).to(device)
    #     with torch.no_grad():
    #         image_features = model.encode_image(data_images).float()
    #         text_features = model.encode_text(texts_tokenized).float()
    #         text_features /= text_features.norm(dim=-1, keepdim=True)
    #         image_features /= image_features.norm(dim=-1, keepdim=True)
    #         text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)#矩阵乘法相当于计算图文的相似度 softmax将相似度转化为概率
    #         top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
    #     for i in range(len(data_texts)):
    #         # print(f"Image Text: {data_texts[i]}")
    #         # print("Predicted Texts:")
    #         label=[data_texts[label_idx] for label_idx in top_labels[i]]
    #         # for label_idx in top_labels[i]:
    #         #     print(data_texts[label_idx])
    #         if data_texts[i] not in label:
    #             count_miss += 1
    #
    #         # print("---------")
    #
    # print(f"Total miss count: {count_miss}")
    # accuracy = 1 - count_miss / len(test_loader.dataset)
    # print(f"Accuracy: {accuracy * 100:.2f}%")
test2_model()
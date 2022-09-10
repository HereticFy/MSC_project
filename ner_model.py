import pandas as pd
from glob import glob
import os
import random
import torch
from torch.utils import data
import numpy as np
import shutil
import os
from tqdm import tqdm
import torch.nn as nn
from torchcrf import CRF
import torch
from transformers import BertModel
from transformers import BertTokenizer

data_file_path = "./r-100/Almond_and_apple_cake_recipe__All_recipes_UK.list"
r_100_path = "./r-100/"
r_200_path = "./r-200/"

def get_annotation(recipe_path):
    with open(recipe_path) as file:
        anns = {}
        for line in file.readlines():
            arr = line.split()
            stpe = arr[0]
            sentence = arr[1]
            start_word = arr[2]
            word = arr[3]
            tag_ner = arr[5]
            df = pd.DataFrame({'word': list(text), 'label': ['O'] * len(text)})
        return anns


def get_ner(recipe_path):
    with open(recipe_path) as file:
        ner_tag = []
        for line in file.readlines():
            arr = line.split()
            tag_ner = arr[-1]
            ner_tag.append(tag_ner)
        return ner_tag

get_ner("./r-100/Almond_and_apple_cake_recipe__All_recipes_UK.list")


def get_tokens(recipe_path):
    with open(recipe_path) as file:
        tokens = []
        for line in file.readlines():
            arr = line.split()
            word = arr[3]
            tokens.append(word)
        return tokens

get_tokens("./r-100/Almond_and_apple_cake_recipe__All_recipes_UK.list")

# Establishing tokens and label correspondence
def generate_annotation(recipe_path):
    for txt_path in glob(recipe_path + '*.list'):
        ners = get_ner(txt_path)
        tokens = get_tokens(txt_path)
        # Create tokens and label correspondence
        df = pd.DataFrame({'word': tokens, 'label': ners})
        # store the file in csv file
        file_name = os.path.split(txt_path)[1][:-4] + ".csv"
        df.to_csv("./annotation/" + file_name, header=None, index=None)

# data_process.py
# split into training set and test set
TRAIN_SAMPLE_PATH = './training_data.txt'
TEST_SAMPLE_PATH = './test_data.txt'


def split_sample(test_size=0.1):
    files = glob("./annotation/" + '*.csv')
    random.seed(0)
    random.shuffle(files)
    n = int(len(files) * test_size)
    test_files = files[:n]
    train_files = files[n:]

    merge_file(train_files, TRAIN_SAMPLE_PATH)
    merge_file(test_files, TEST_SAMPLE_PATH)

def merge_file(files, target_path):
    with open(target_path, 'a') as file:
        for f in files:
            text = open(f).read()
            file.write(text)


VOCAB_PATH = './vocab.txt'
LABEL_PATH = './label.txt'

WORD_PAD = '<PAD>'
WORD_UNK = '<UNK>'

VOCAB_SIZE = 2407



def generate_vocab():
    df = pd.read_csv(TRAIN_SAMPLE_PATH, usecols=[0], names=['word'])
    vocab_list = [WORD_PAD, WORD_UNK] + df['word'].value_counts().keys().tolist()
    vocab_list = vocab_list[:VOCAB_SIZE]
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab = pd.DataFrame(list(vocab_dict.items()))
    vocab.to_csv(VOCAB_PATH, header=None, index=None)


def generate_label():
    df = pd.read_csv(TRAIN_SAMPLE_PATH, usecols=[1], names=['label'])
    label_list = df['label'].value_counts().keys().tolist()
    label_dict = {v: k for k, v in enumerate(label_list)}
    label = pd.DataFrame(list(label_dict.items()))
    label.to_csv(LABEL_PATH, header=None, index=None)


def get_vocab():
    df = pd.read_csv(VOCAB_PATH, names=['word', 'id'])
    return list(df['word']), dict(df.values)

def get_label():
    df = pd.read_csv(LABEL_PATH, names=['label', 'id'])
    return list(df['label']), dict(df.values)


WORD_PAD_ID = 0
WORD_UNK_ID = 1
LABEL_O_ID = 0

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    max_len = len(batch[0][0])
    input = []
    target = []
    mask = []
    for item in batch:
        pad_len = max_len - len(item[0])
        input.append(item[0] + [WORD_PAD_ID] * pad_len)
        target.append(item[1] + [LABEL_O_ID] * pad_len)
        mask.append([1] * len(item[0]) + [0] * pad_len)
    return torch.tensor(input), torch.tensor(target), torch.tensor(mask).bool()

VOCAB_SIZE = 2407
EMBEDDING_DIM = 100
HIDDEN_SIZE = 256
TARGET_SIZE = 21
LR = 1e-3
EPOCH = 500

MODEL_DIR = './model/'

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch.nn as nn

bert_model = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model)
VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'Ac-B', 'Ac-I', 'Ac2-B', 'Ac2-I', 'Af-B', 'Af-I', 'At-B', 'At-I',
       'D-B', 'D-I', 'F-B', 'F-I', 'O', 'Q-B', 'Q-I', 'Sf-B', 'Sf-I',
       'St-B', 'St-I', 'T-B', 'T-I')

tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
MAX_LEN = 64-2

class NerDataset(Dataset):
    ''' Generate our dataset '''
    def __init__(self, f_path):
        self.sents = []
        self.tags_li = []

        with open(f_path, 'r', encoding='utf-8') as f:
            lines = [line.split('\n')[0] for line in f.readlines() if len(line.strip())!=0]
          
        tags =  [line.split(',')[1] for line in lines]
        for i in range(len(tags)):
            if tags[i] == '"':
                tags[i] = 'O'
        words = [line.split(',')[0] for line in lines]

        word, tag = [], []
        for char, t in zip(words, tags):
            if char == '"':
                word.append(char)
                tag.append('O')
                
            if char != '.':
                word.append(char)
                tag.append(t)
            else:
                if len(word) > MAX_LEN:
                    self.sents.append(['[CLS]'] + word[:MAX_LEN] + ['[SEP]'])
                    self.tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
                else:
                    self.sents.append(['[CLS]'] + word + ['[SEP]'])
                    self.tags_li.append(['[CLS]'] + tag + ['[SEP]'])
                word, tag = [], []

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]
        token_ids = tokenizer.convert_tokens_to_ids(words)
        laebl_ids = [tag2idx[tag] for tag in tags]
        seqlen = len(laebl_ids)
        return token_ids, laebl_ids, seqlen

    def __len__(self):
        return len(self.sents)

def PadBatch(batch):
    maxlen = max([i[2] for i in batch])
    token_tensors = torch.LongTensor([i[0] + [0] * (maxlen - len(i[0])) for i in batch])
    label_tensors = torch.LongTensor([i[1] + [0] * (maxlen - len(i[1])) for i in batch])
    mask = (token_tensors > 0)
    return token_tensors, label_tensors, mask


class Bert_BiLSTM_CRF(nn.Module):

    def __init__(self, tag_to_ix, embedding_dim=768, hidden_dim=256):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(hidden_dim * 2, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)
    
    def _get_features(self, sentence):
        #print(sentence.shape)
        with torch.no_grad():
            embeds  = self.bert(sentence)['last_hidden_state']
        enc, _ = self.lstm(embeds)
        #enc = self.dropout(enc)
        feats = self.linear(enc)
        return feats


    def forward(self, input, mask):
        out = self._get_features(input)

        return self.crf.decode(out, mask)
    
    
    def loss_fn(self, input, target, mask):
        y_pred = self._get_features(input)
        return -self.crf.forward(y_pred, target, mask).mean()

model = Bert_BiLSTM_CRF(tag2idx)
print(model)

dataset = NerDataset(TRAIN_SAMPLE_PATH)
loader = data.DataLoader(
    dataset,
    batch_size=10,
    shuffle=False,
    collate_fn=PadBatch,
)

model = Bert_BiLSTM_CRF(tag2idx)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for e in range(100):
    for b, (input, target, mask) in enumerate(loader):
        y_pred = model(input, mask)
        loss = model.loss_fn(input, target, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 10 == 0:
            print('>> epoch:', e, 'loss:', loss.item())

    torch.save(model, "bert_model/" + f'model_{e}.pth')

dataset = NerDataset(TEST_SAMPLE_PATH)
loader = data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=PadBatch,
)

with torch.no_grad():
    model = torch.load("bert_model/" + 'model_99.pth')
    y_true_list_test = []
    y_pred_list_test = []
    for b, (input, target, mask) in enumerate(loader):
        y_pred = model(input, mask)
        loss = model.loss_fn(input, target, mask)
        for lst in y_pred:
            y_pred_list_test += lst
        for y,m in zip(target, mask):
            y_true_list_test += y[m==True].tolist()
        #print('>> batch:', b, 'loss:', loss.item())
    y_true_tensor_test = torch.tensor(y_true_list_test)
    y_pred_tensor_test = torch.tensor(y_pred_list_test)
    accuracy = (y_true_tensor_test == y_pred_tensor_test).sum() / len(y_true_tensor_test)
    print('>> total:', len(y_true_tensor_test), 'accuracy:', accuracy.item())

dataset = NerDataset(TRAIN_SAMPLE_PATH)
loader = data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=PadBatch,
)

with torch.no_grad():
    model = torch.load("bert_model/" + 'model_99.pth')
    y_true_list_train = []
    y_pred_list_train = []
    for b, (input, target, mask) in enumerate(loader):
        y_pred = model(input, mask)
        loss = model.loss_fn(input, target, mask)
        for lst in y_pred:
            y_pred_list_train += lst
        for y,m in zip(target, mask):
            y_true_list_train += y[m==True].tolist()
        #print('>> batch:', b, 'loss:', loss.item())
    y_true_tensor_train = torch.tensor(y_true_list_train)
    y_pred_tensor_train = torch.tensor(y_pred_list_train)
    accuracy = (y_true_tensor_train == y_pred_tensor_train).sum()/len(y_true_tensor_train)
    print('>> total:', len(y_true_tensor_train), 'accuracy:', accuracy.item())


y_true_tensor = torch.cat((y_true_tensor_train, y_true_tensor_test))
y_pred_tensor = torch.cat((y_pred_tensor_train, y_pred_tensor_test))


TP_l = []
FP_l = []
FN_l = []

for k in range(24):
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(y_true_tensor)):
        if y_true_tensor[i] == k and y_pred_tensor[i] == k:
            TP += 1
        if y_true_tensor[i] == k and y_pred_tensor[i] != k:
            FN += 1
        if y_true_tensor[i] != k and y_pred_tensor[i] == k:
            FP += 1
    TP_l.append(TP)
    FP_l.append(FP)
    FN_l.append(FN)
    
precisions = []
recalls = []

for i in range(24):
    precisions.append(TP_l[i] / (TP_l[i] + FP_l[i] + 1))
    recalls.append(TP_l[i] / (TP_l[i] + FN_l[i] + 1))

F1_score = []
for i in range(1, len(precisions)):
    F1_score.append(2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i]))
    
F1_score

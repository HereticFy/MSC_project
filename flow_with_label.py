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

TRAIN_FLOW_PATH = "/content/drive/MyDrive/recipe_data/edge_training_data.txt"
TEST_FLOW_PATH = "/content/drive/MyDrive/recipe_data/edge_test_data.txt"



def get_flow_annotation(recipe_path):
    with open(recipe_path, encoding='utf-8', errors='ignore') as file:
        res = []
        for line in file.readlines():
            arr = line.split()
            res.append(arr)
        return res

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
            print(arr)
        return anns

def get_in_position(recipe_path):
    with open(recipe_path, encoding='utf-8', errors='ignore') as file:
        in_positions = []
        for line in file.readlines():
            arr = line.split()
            if len(arr) > 1 and '#' not in arr[0] and arr[3] != "-":
                in_position = arr[:3]
                in_positions.append(in_position)
        return in_positions

def get_out_position(recipe_path):
    with open(recipe_path, encoding='utf-8', errors='ignore') as file:
        out_positions = []
        for line in file.readlines():
            arr = line.split()
            if len(arr) > 1 and '#' not in arr[0] and arr[3] != "-":
                out_position = arr[4:]
                out_positions.append(out_position)
        return out_positions

def get_tags(recipe_path):
    with open(recipe_path, encoding='utf-8', errors='ignore') as file:
        tags = []
        for line in file.readlines():
            arr = line.split()
            if len(arr) > 1 and '#' not in arr[0] and arr[3] != "-":
                tag = arr[3]
                if tag == "s":
                    tag = "d"
                if tag == "v":
                    tag = "v-tm"
                tags.append(tag)
        return tags

def get_recipe(recipe_path):
    with open(recipe_path) as file:
        res = []
        for line in file.readlines():
            arr = line.split()
            res.append(arr)
    return res



def generate_text_file(recipe_path):
    for txt_path in glob(recipe_path + '*.list'):
        flag = ["1", "1"]
        sentence = ""
        inst, sentence_nb, sentence_text = [], [], []
        data = get_recipe(txt_path)
        for t in data:
            if t[:2] == flag:
                if t[3] != ".":
                    sentence += t[3] + " "
                else:
                    sentence = sentence[:-1] + "."
            else:
                inst.append(flag[0])
                sentence_nb.append(flag[1])
                sentence_text.append(sentence)
                flag = t[:2]
                sentence = t[3] + " "
        inst.append(flag[0])
        sentence_nb.append(flag[1])
        sentence_text.append(sentence)
        # Create a csv file including the full sentences of a recipe
        df = pd.DataFrame({'instruction': inst, 'sentence_nb': sentence_nb, "sentence": sentence_text})
        # store the file in csv file
        file_name = os.path.split(txt_path)[1][:-4] + "csv"
        df.to_csv(recipe_path + file_name, header=None, index=None)


def get_text(recipe_path):
    df = pd.read_csv(recipe_path, header = None)
    instruction = df[0].tolist()
    sentence_nb = df[1].tolist()
    text = df[2].tolist()
    return instruction, sentence_nb, text

# Establishing tokens and label correspondence
def generate_flow_annotation(recipe_path, label_list):
    for txt_path in glob(recipe_path + '*.flow'):
        tags = get_tags(txt_path)
        in_positions = get_in_position(txt_path)
        out_positions = get_out_position(txt_path)
        recipe = get_recipe(txt_path[:-4] + "list")
        instruction, sentence_nb, text = get_text(txt_path[:-4] + "csv")
        # e_1 points to e_2, e1 is the first entity appers in the sentence, e2 is the second
        # And creating new labels for the dataset
        nodes_in, nodes_out, e_1, e_2, point_relation, labels = [], [], [], [], [], []
        
        for i in range(len(in_positions)):
            for j in range(len(recipe)):
                if in_positions[i] == recipe[j][:3]:
                    nodes_in.append(recipe[j])
                    in_entity = recipe[j][3]
                    #e_1.append(recipe[j][3])
                if out_positions[i] == recipe[j][:3]:
                    nodes_out.append(recipe[j])
                    out_entity = recipe[j][3]
                    #e_2.append(recipe[j][3])
            point_relation.append((in_entity, out_entity))
            if in_positions[i] < out_positions[i]:
                e_1.append(in_entity)
                e_2.append(out_entity)
                labels.append(tags[i] + "(e1,e2)")
            else:
                e_1.append(out_entity)
                e_2.append(in_entity)
                labels.append(tags[i] + "(e2,e1)")
                
        # adding a new label called not-relate(e1, e2) and not-relate(e2, e1) for the pair of 
        # entities that has no relationship in the flow chart
        for r in range(len(recipe)):
            for e in range(len(recipe)):
                # entity with label "O" is not in the flow chat(verified)
                if recipe[r][:3] not in in_positions and recipe[e][:3] not in out_positions and recipe[r][-1] != "O" and recipe[e][-1] != "O":
                    in_positions.append(recipe[r][:3])
                    nodes_in.append(recipe[r])
                    out_positions.append(recipe[e][:3])
                    nodes_out.append(recipe[e])
                    if recipe[r][:3] > recipe[e][:3]:
                        e_1.append(recipe[e][3])
                        e_2.append(recipe[r][3])
                        labels.append("not-relate(e2, e1)")
                    else:
                        e_2.append(recipe[e][3])
                        e_1.append(recipe[r][3])
                        labels.append("not-relate(e1, e2)")
                if recipe[r][:3] not in out_positions and recipe[e][:3] not in in_positions and recipe[r][-1] != "O" and recipe[e][-1] != "O":
                    out_positions.append(recipe[r][:3])
                    nodes_out.append(recipe[r])
                    in_positions.append(recipe[e][:3])
                    nodes_in.append(recipe[e])
                    if recipe[r][:3] > recipe[e][:3]:
                        e_1.append(recipe[e][3])
                        e_2.append(recipe[r][3])
                        labels.append("not-relate(e1, e2)")
                    else:
                        e_2.append(recipe[e][3])
                        e_1.append(recipe[r][3])
                        labels.append("not-relate(e2, e1)")
                        
        # convert the string type labels to int type for the classification
        all_class = list(set(labels))
        label_nb = []
        for i in range(len(in_positions)):
            label_nb.append(label_list.index(labels[i])) # index of label
        
                                    
        in_pos, out_pos, in_nodes, out_nodes = [], [], [], []
        
        for k in range(len(in_positions)):
            in_pos.append(",".join(in_positions[k]))
            out_pos.append(",".join(out_positions[k]))
            in_nodes.append(",".join(nodes_in[k]))
            out_nodes.append(",".join(nodes_out[k]))
            
        sentence_related = []
        for t in range(len(in_positions)):
            # if 2 entities in the same sentence
            sentence = ""
            if in_positions[t][:2] == out_positions[t][:2]:
                for i in range(len(instruction)):
                    if int(in_positions[t][0]) == instruction[i] and int(in_positions[t][1]) == sentence_nb[i]:
                        sentence = text[i]
                        
                #sentence_related.append(sentence.strip())
                sentence_related.append(sentence)

            # if 2 entities in different sentences
            else:
                first_sentence = ""
                second_sentence = ""
                for j in range(len(instruction)):
                    if int(in_positions[t][0]) == instruction[j] and int(in_positions[t][1]) == sentence_nb[j]:
                        first_sentence = text[j]
                    if int(out_positions[t][0]) == instruction[j] and int(out_positions[t][1]) == sentence_nb[j]:
                        second_sentence = text[j]
                #sentence_related.append(first_sentence[:-1].strip() + " " + second_sentence.strip())
                sentence_related.append(first_sentence + " " + second_sentence)
                    
        # Create tokens and label correspondence
        df = pd.DataFrame({'in_positions': in_pos, 'out_positions': out_pos, "labels": labels,"label_nb":label_nb, "nodes_in": in_nodes, "nodes_out": out_nodes, "entity_1": e_1, "entity_2": e_2, "sentence_related": sentence_related})
        # store the file in csv file
        file_name = os.path.split(txt_path)[1][:-4] + "csv"
        df.to_csv("./edge_dataset/" + file_name, header=None, index=None)
        


edge_train_csv = pd.read_csv(TRAIN_FLOW_PATH, header=None)
edge_test_csv = pd.read_csv(TEST_FLOW_PATH, header=None)


# number of different types of labels
#EDGE_LABELS_NUMBER = len(EDGE_LABEL_LIST)

#TRAIN_SENTENCES_LIST = edge_train_csv[8].tolist()
train_sentence = edge_train_csv[8].tolist()
TRAIN_SENTENCES_LIST = []
for s in train_sentence:
  TRAIN_SENTENCES_LIST.append(s.replace(".", "") + ".")



TRAIN_LABELS = edge_train_csv[3].tolist()

train_e1 = edge_train_csv[6].tolist()
train_e2 = edge_train_csv[7].tolist()
TRAIN_ENTITIES = []
for i in range(len(train_e1)):
    TRAIN_ENTITIES.append(train_e1[i] + " " + train_e2[i])

test_sentence = edge_test_csv[8].tolist()
TEST_SENTENCES_LIST = []

for s in test_sentence:
  TEST_SENTENCES_LIST.append(s.replace(".", "") + ".")

TEST_LABELS = edge_test_csv[3].tolist()

test_e1 = edge_test_csv[6].tolist()
test_e2 = edge_test_csv[7].tolist()
TEST_ENTITIES = []
for i in range(len(test_e1)):
    TEST_ENTITIES.append(test_e1[i] + " " + test_e2[i] + ".")

NUMBER_CLASSES = len(set(TRAIN_LABELS + TEST_LABELS))

label_list = ['a(e2,e1)',
 'not-relate(e1, e2)',
 't(e2,e1)',
 't(e1,e2)',
 't-eq(e2,e1)',
 'a(e1,e2)',
 'f-eq(e1,e2)',
 'v-tm(e2,e1)',
 't-part-of(e1,e2)',
 'o(e1,e2)',
 't-part-of(e2,e1)',
 't-eq(e1,e2)',
 'not-relate(e2, e1)',
 'd(e1,e2)',
 'f-part-of(e2,e1)',
 'f-eq(e2,e1)',
 'f-comp(e2,e1)',
 'a-eq(e1,e2)',
 'f-set(e1,e2)',
 'v-tm(e1,e2)',
 'f-part-of(e1,e2)',
 'a-eq(e2,e1)',
 'f-comp(e1,e2)',
 'd(e2,e1)',
 't-comp(e1,e2)',
 'f-set(e2,e1)',
 'o(e2,e1)',
 't-comp(e2,e1)']

from transformers import BertTokenizer
import torch.utils.data as Data
from transformers import BertForSequenceClassification

from torch.optim import AdamW

def convert(entities, sentences, target): # name_list, sentence_list, label_list
    input_ids, token_type_ids, attention_mask = [], [], []
    for i in range(len(sentences)):
        encoded_dict = tokenizer.encode_plus(
            entities[i] + " " + sentences[i],        # 输入文本
            add_special_tokens = True,      # 添加 '[CLS]' 和 '[SEP]'
            max_length = 128,             # 不够填充
            padding = 'max_length',
            #truncation = True,             # 太长截断
            return_tensors = 'pt',         # 返回 pytorch tensors 格式的数据
        )
        input_ids.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict['token_type_ids'])
        attention_mask.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)

    input_ids = torch.LongTensor(input_ids)
    token_type_ids = torch.LongTensor(token_type_ids)
    attention_mask = torch.LongTensor(attention_mask)
    target = torch.LongTensor(target)

    return input_ids, token_type_ids, attention_mask, target


input_ids, token_type_ids, attention_mask, target = convert(TRAIN_ENTITIES, TRAIN_SENTENCES_LIST, TRAIN_LABELS)

import torch.utils.data as Data
#from transformers import XLNetForSequenceClassification
from torch.optim import AdamW

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten() # [3, 5, 8, 1, 2, ....]
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train_eval(model, input_ids, token_type_ids, attention_mask, labels):
    batch_size = 100
    train_data = Data.TensorDataset(input_ids, token_type_ids, attention_mask, labels)
    train_dataloader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    for e in tqdm(range(10)):

        for i, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            loss = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2], labels=batch[3])[0]
            print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred_label = pred(model)
        acc = 0
        for i in range(len(pred_label)):
          if pred_label[i] == TEST_LABELS[i]:
              acc += 1
        print("acc:")
        print(acc/len(pred_label))
        
    torch.save(model, "/content/drive/MyDrive/recipe_data/edge_models/" + 'edge_model_1002.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=NUMBER_CLASSES).to(device)
#model = torch.load("/content/drive/MyDrive/recipe_data/edge_models/edge_model_9.pth")
train_eval(model, input_ids, token_type_ids, attention_mask, target)

TEST_CLASSES_LIST = list(set(TRAIN_LABELS + TEST_LABELS))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pred(model_path):
    # load model
    model = torch.load(model_path)

    sentence_list = []

    input_ids, token_type_ids, attention_mask, _ = convert(TEST_ENTITIES, TEST_SENTENCES_LIST, TEST_LABELS) # whatever name_list and label_list
    dataset = Data.TensorDataset(input_ids, token_type_ids, attention_mask)
    loader = Data.DataLoader(dataset, 1, False)

    pred_label = []
    model.eval()
    for i, batch in enumerate(loader):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            #logits = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2])[0]
            logits = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2]).logits
            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()
            pred_label.extend(preds)
    
    #for i in range(len(pred_label)):
    #    pred_label[i] = TEST_CLASSES_LIST[pred_label[i]]
    
    #pd.DataFrame(data=pred_label, index=range(len(pred_label))).to_csv('pred.csv')
    return pred_label


def pred(model):
    # load model
    #model = torch.load(model_path)

    sentence_list = []

    input_ids, token_type_ids, attention_mask, _ = convert(TEST_ENTITIES, TEST_SENTENCES_LIST, TEST_LABELS) # whatever name_list and label_list
    dataset = Data.TensorDataset(input_ids, token_type_ids, attention_mask)
    loader = Data.DataLoader(dataset, 1, False)

    pred_label = []
    model.eval()
    for i, batch in enumerate(loader):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            #logits = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2])[0]
            logits = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2]).logits
            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()
            pred_label.extend(preds)
    
    #for i in range(len(pred_label)):
    #    pred_label[i] = TEST_CLASSES_LIST[pred_label[i]]
    
    #pd.DataFrame(data=pred_label, index=range(len(pred_label))).to_csv('pred.csv')
    return pred_label





def transfer_orignal_label(target_list):
  result = []
  for i in range(len(target_list)):

    if target_list[i] == 0 or target_list[i] == 5:
      result.append(0)
    elif target_list[i] == 2 or target_list[i] == 3:
      result.append(1)
    elif target_list[i] == 13 or target_list[i] == 23:
      result.append(2)
    elif target_list[i] == 24 or target_list[i] == 27:
      result.append(3)

    elif target_list[i] == 16 or target_list[i] == 22:
      result.append(4)

    elif target_list[i] == 6 or target_list[i] == 15:
      result.append(5)

    elif target_list[i] == 14 or target_list[i] == 20:
      result.append(6)

    elif target_list[i] == 18 or target_list[i] == 25:
      result.append(7)

    elif target_list[i] == 4 or target_list[i] == 11:
      result.append(8)

    elif target_list[i] == 8 or target_list[i] == 10:
      result.append(9)

    elif target_list[i] == 21 or target_list[i] == 17:
      result.append(10)

    elif target_list[i] == 7 or target_list[i] == 19:
      result.append(11)

    elif target_list[i] == 9 or target_list[i] == 26:
      result.append(12)
    else:
      result.append(13)
  return result

def pred_tt(model):
    # load model
    #model = torch.load(model_path)

    sentence_list = []

    input_ids, token_type_ids, attention_mask, _ = convert(enti, se, ll) # whatever name_list and label_list
    dataset = Data.TensorDataset(input_ids, token_type_ids, attention_mask)
    loader = Data.DataLoader(dataset, 1000, False)

    pred_label = []
    model.eval()
    for i, batch in enumerate(loader):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            #logits = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2])[0]
            logits = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2]).logits
            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()
            pred_label.extend(preds)
    
    #for i in range(len(pred_label)):
    #    pred_label[i] = TEST_CLASSES_LIST[pred_label[i]]
    
    #pd.DataFrame(data=pred_label, index=range(len(pred_label))).to_csv('pred.csv')
    return pred_label

def pred_train(model):
    # load model
    #model = torch.load(model_path)

    sentence_list = []

    input_ids, token_type_ids, attention_mask, _ = convert(TRAIN_ENTITIES, TRAIN_SENTENCES_LIST, TRAIN_LABELS) # whatever name_list and label_list
    dataset = Data.TensorDataset(input_ids, token_type_ids, attention_mask)
    loader = Data.DataLoader(dataset, 1000, False)

    pred_label = []
    model.eval()
    for i, batch in enumerate(loader):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            #logits = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2])[0]
            logits = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2]).logits
            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()
            pred_label.extend(preds)
    
    #for i in range(len(pred_label)):
    #    pred_label[i] = TEST_CLASSES_LIST[pred_label[i]]
    
    #pd.DataFrame(data=pred_label, index=range(len(pred_label))).to_csv('pred.csv')
    return pred_label

total_true = transfer_orignal_label(TRAIN_LABELS + TEST_LABELS)
total_pred = transfer_orignal_label(pred_train_label + pred_label)


TP_l = []
FP_l = []
FN_l = []

for k in range(13):
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(total_true)):
        if total_true[i] == k and total_pred[i] == k:
            TP += 1
        elif total_true[i] == k and total_pred[i] != k:
            FN += 1
        elif total_true[i] != k and total_pred[i] == k:
            FP += 1
    TP_l.append(TP)
    FP_l.append(FP)
    FN_l.append(FN)

precisions = []
recalls = []

for i in range(13):
    precisions.append(TP_l[i] / (TP_l[i] + FP_l[i]))
    recalls.append(TP_l[i] / (TP_l[i] + FN_l[i]))

F1_score = []
for i in range(0, len(precisions)):
    F1_score.append(2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i]))

F1_score

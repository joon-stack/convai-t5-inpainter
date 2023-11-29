import json

from itertools import chain

from transformers import TapasConfig, TapasForQuestionAnswering, AdamW, TapasTokenizer, BertTokenizer, AutoTokenizer, AutoModel
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import pandas as pd

import random
import numpy as np
import os

from tqdm import tqdm, trange

seed = 2023

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print('pwd', os.getcwd())


def tokenize(x):
    context_tokenized = tokenizer(x['context'],  padding=True, truncation=True, max_length=512, return_tensors='pt')
    target_tokenized = tokenizer(x['target'], padding=True, truncation=True, max_length=512, return_tensors='pt')
    return {
        "context": context_tokenized,
        "target": target_tokenized,
    }

class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.context = data_dict['context']
        self.target = data_dict['target']

    def __len__(self):
        return self.context['input_ids'].shape[0]

    def __getitem__(self, idx):
        context_input_ids = self.context['input_ids'][idx]
        context_attention_mask = self.context['attention_mask'][idx]
        target_input_ids = self.target['input_ids'][idx]
        target_attention_mask = self.target['attention_mask'][idx]

        return context_input_ids, context_attention_mask, target_input_ids, target_attention_mask
    

def inference():
    model_q = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    model_t = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    model_q.load_state_dict(torch.load("checkpoints/model_q_epoch_194.pt")['model_state_dict'])
    model_t.load_state_dict(torch.load("checkpoints/model_t_epoch_194.pt")['model_state_dict'])
    print("Model load success from ", "checkpoints/"+args.checkpoint_name)
    model_q.to('cuda:0')
    model_t.to('cuda:0')

    for batch in inf_dataloader:
        input_ids_q, attention_mask_q,  input_ids_t, attention_mask_t = batch
        input_ids_q = input_ids_q.to(f'cuda:{model_q.device_ids[0]}')
        attention_mask_q = attention_mask_q.to(f'cuda:{model_q.device_ids[0]}')
        input_ids_t = input_ids_t.to(f'cuda:{model_t.device_ids[0]}')
        attention_mask_t = attention_mask_t.to(f'cuda:{model_t.device_ids[0]}')
        h_t = model_t(
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
        ).pooler_output

        # B*512 shape
        h_q = model_q(
            input_ids=input_ids_q,
            attention_mask=attention_mask_q
        ).pooler_output

        logits = torch.softmax(h_q @ h_t.T, dim=1)
        labels = torch.eye(logits.shape[0]).to(f'cuda:{model_q.device_ids[0]}')

        size = len(input_ids_q)
        retrieved_top1 = torch.topk(logits, 1)
        retrieved_top5 = torch.topk(logits, 5)
        retrieved_top10 = torch.topk(logits, 10)

        count_top1 = 0
        count_top5 = 0
        count_top10 = 0

        for i, row in enumerate(retrieved_top1):
            if i in row:
                count_top1 += 1
        
        for i, row in enumerate(retrieved_top5):
            if i in row:
                count_top5 += 1

        for i, row in enumerate(retrieved_top10):
            if i in row:
                count_top10 += 1
        
        recall_top1 = count_top1 / size
        recall_top5 = count_top5 / size
        recall_top10 =count_top10 / size

        print(f"R@1: {recall_top1:.3f}, R@5: {recall_top5:.3f}, R@10: {recall_top10:.3f}")





    

# Checkpoint directory
os.mkdir("checkpoints") if not os.path.exists("checkpoints") else None
# check GPU
print('Cuda:', torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



with open("data/hybrid/experimental_data.json", 'r') as f:
    hybrid = json.load(f)

convs = list(hybrid['conversations'].values())
queries = list(map(lambda x: list(map(lambda y: hybrid['qas'][y]['current_query'], x)), convs))
responses = list(map(lambda x: list(map(lambda y: hybrid['qas'][y]['long_response_to_query'], x)), convs))
retrieved = list(map(lambda x: list(map(lambda y: hybrid['qas'][y]['correct_next_cands_ids'][0], x)), convs))
retrieved_inputs = list(map(lambda x: list(map(lambda y: hybrid['all_candidates'][y]['linearized_input'], x)), retrieved))
output = []
for x, i in enumerate(responses):
    tmp = []
    tmp.append("0: " + queries[x][0] + " 1: [MASK]")
    for j in range(1, len(i)):
        tmp.append(tmp[j-1][:-6] + responses[x][j-1] + " 0: " + queries[x][j] + " 1: [MASK]")
    output.append(tmp)


queries_flat = list(chain(*queries))
output_flat = list(chain(*output))
retrieved_inputs_flat = list(chain(*retrieved_inputs))
retrieved_inputs_filtered = list(filter(lambda x: x[1:3] == 'TA' or x[1:3] == 'PA', retrieved_inputs_flat))
output_filtered = [output_flat[i] for (i, x) in enumerate(retrieved_inputs_flat) if  x[1:3] == 'TA' or x[1:3] == 'PA']
retrieved_inputs_filtered = list(map(lambda x: x.replace("$", "").strip(), retrieved_inputs_filtered))
output_filtered_trn = output_filtered[:int(0.8*len(output_filtered))]
retrieved_inputs_filtered_trn = retrieved_inputs_filtered[:int(0.8*len(output_filtered))]
output_filtered_val = output_filtered[int(0.8*len(output_filtered)):]
retrieved_inputs_filtered_val = retrieved_inputs_filtered[int(0.8*len(output_filtered)):]
data_dict_trn = {'context': output_filtered_trn, 'target': retrieved_inputs_filtered_trn}
data_dict_val = {'context': output_filtered_val, 'target': retrieved_inputs_filtered_val}

tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
tokenized_ds_trn = tokenize(data_dict_trn)
ds_trn = CustomDataset(tokenized_ds_trn)
tokenized_ds_val = tokenize(data_dict_val)
ds_val = CustomDataset(tokenized_ds_val)

batch_size = 512

train_dataloader = DataLoader(ds_trn, batch_size=batch_size)
val_dataloader = DataLoader(ds_val, batch_size=batch_size)
inf_dataloader = DataLoader(ds_val, batch_size=len(ds_val))


model_q = AutoModel.from_pretrained('prajjwal1/bert-tiny')
model_t = AutoModel.from_pretrained('prajjwal1/bert-tiny')
optim_q = AdamW(model_q.parameters(), lr=5e-5)
optim_t = AdamW(model_t.parameters(), lr=5e-5)

# for multiple GPUs
# model_q = nn.DataParallel(model_q, device_ids = [0, 1, 2, 3])
# model_t = nn.DataParallel(model_t, device_ids = [0, 1, 2, 3])
# model_q.to(f'cuda:{model_q.device_ids[0]}')
# model_t.to(f'cuda:{model_t.device_ids[0]}')

model_q.to('cuda:0')
model_t.to('cuda:0')


criterion = nn.CrossEntropyLoss()
for epoch in range(1, 1001):
    loss_batch = 0
    loss_val = 0
    best_loss = np.inf
    model_q.train()
    model_t.train()

    model_q.eval()
    model_t.eval()
    recall_top1 = 0
    recall_top5 = 0
    recall_top10 = 0
    

    for batch in train_dataloader:
        input_ids_q, attention_mask_q,  input_ids_t, attention_mask_t= batch
        # for multiple GPUs
        # input_ids_q = input_ids_q.to(f'cuda:{model_q.device_ids[0]}')
        # attention_mask_q = attention_mask_q.to(f'cuda:{model_q.device_ids[0]}')
        # input_ids_t = input_ids_t.to(f'cuda:{model_t.device_ids[0]}')
        # attention_mask_t = attention_mask_t.to(f'cuda:{model_t.device_ids[0]}')

        input_ids_q = input_ids_q.to('cuda:0')
        attention_mask_q = attention_mask_q.to('cuda:0')
        input_ids_t = input_ids_t.to('cuda:0')
        attention_mask_t = attention_mask_t.to('cuda:0')

        optim_q.zero_grad()
        optim_t.zero_grad()

        # B*512 shape
        h_t = model_t(
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
        ).pooler_output

        # B*512 shape
        h_q = model_q(
            input_ids=input_ids_q,
            attention_mask=attention_mask_q
        ).pooler_output


        # B*B shape
        logits = torch.softmax(h_q @ h_t.T, dim=1)
        labels = torch.eye(logits.shape[0]).to('cuda:0')
        loss = criterion(logits, labels)
        loss.backward()
        optim_q.step()
        optim_t.step()
        loss_batch += loss.mean().item()
    loss_batch /= batch_size
    # print(f"Epoch {epoch}: {loss_batch:.4f}")

    for batch in val_dataloader:
        input_ids_q, attention_mask_q,  input_ids_t, attention_mask_t = batch
        input_ids_q = input_ids_q.to('cuda:0')
        attention_mask_q = attention_mask_q.to('cuda:0')
        input_ids_t = input_ids_t.to('cuda:0')
        attention_mask_t = attention_mask_t.to('cuda:0')

        optim_q.zero_grad()
        optim_t.zero_grad()

        # B*512 shape
        h_t = model_t(
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
        ).pooler_output

        # B*512 shape
        h_q = model_q(
            input_ids=input_ids_q,
            attention_mask=attention_mask_q
        ).pooler_output


        # B*B shape
        logits = torch.softmax(h_q @ h_t.T, dim=1)
        # for multiple GPUs
        # labels = torch.eye(logits.shape[0]).to(f'cuda:{model_q.device_ids[0]}')
        labels = torch.eye(logits.shape[0]).to(f'cuda:0')
        loss = criterion(logits, labels)
        loss_val += loss.mean().item()

        size = len(input_ids_q)
        retrieved_top1 = torch.topk(logits, 1)

        retrieved_top5 = torch.topk(logits, 5)
        retrieved_top10 = torch.topk(logits, 10)



        count_top1 = 0
        count_top5 = 0
        count_top10 = 0

        for i, row in enumerate(retrieved_top1.indices):
            if i in row:
                count_top1 += 1
        
        for i, row in enumerate(retrieved_top5.indices):
            if i in row:
                count_top5 += 1

        for i, row in enumerate(retrieved_top10.indices):
            if i in row:
                count_top10 += 1
        
        recall_top1 += count_top1 / size
        recall_top5 += count_top5 / size
        recall_top10 += count_top10 / size

    

        


        
    loss_val /= len(val_dataloader)
    recall_top1 /= len(val_dataloader)
    recall_top5 /= len(val_dataloader)
    recall_top10 /= len(val_dataloader)
    print(f"epoch: {epoch}, loss:{loss_val}, R@1: {recall_top1:.3f}, R@5: {recall_top5:.3f}, R@10: {recall_top10:.3f}")



    if loss_val < best_loss:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model_q.state_dict(),
                'optimizer_state_dict': optim_q.state_dict(),
                'val_loss': best_loss,
                },
                f"checkpoints/model_q_epoch_{epoch}.pt"
            )
        torch.save({
                'epoch': epoch,
                'model_state_dict': model_t.state_dict(),
                'optimizer_state_dict': optim_t.state_dict(),
                'val_loss': best_loss,
                },
                f"checkpoints/model_t_epoch_{epoch}.pt"
            )
    
        
 
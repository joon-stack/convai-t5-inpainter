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
from datetime import datetime

import argparse

seed = 2023

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print('pwd', os.getcwd())


def generate_datetime_key():
    """
    Generate a key based on the current date and time.
    """
    now = datetime.now()
    return now.strftime("%m%d%H%M")

def create_run_directory(base_directory, key):
    """
    Create a directory based on the random key under the specified base directory.
    """
    run_directory = os.path.join(base_directory, key)
    os.mkdir(run_directory) if not os.path.exists(run_directory) else None
    return run_directory
    
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
    
def flatten_table_hybrid(x):
    title = x['section_title']
    header = x['header']
    header_flat = list(map(lambda y: y[0], header))
    header_flat = " ".join(header_flat)
    data = x['data']
    data_flat = list(map(lambda z: list(map(lambda y: " ".join(y[:-1]), z)), data))
    data_flat = list(map(lambda x: " ".join(x), data_flat))
    data_flat = " ".join(data_flat)
    result = title + " " + header_flat + " " + data_flat
    return result

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
        recall_top10 = count_top10 / size

        print(f"R@1: {recall_top1:.3f}, R@5: {recall_top5:.3f}, R@10: {recall_top10:.3f}")





    

# Checkpoint directory




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment_name", type=str, default=None, help="the name of augmented dataset, None: no augment")
    args = parser.parse_args()

    os.mkdir("checkpoints") if not os.path.exists("checkpoints") else None
    base_directory = "checkpoints"
    random_key = generate_datetime_key()
    run_directory = create_run_directory(base_directory, random_key)
    print(f"Random Key: {random_key}")
    print(f"Run Directory: {run_directory}")
    # check GPU
    print('Cuda:', torch.cuda.is_available())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with open("data/hybrid/experimental_data.json", 'r') as f:
        hybrid = json.load(f)

    with open("data/hybrid/traindev_tables.json", 'r') as f:
        tables = json.load(f)

    if args.augment_name:
        with open(f"data/retrieval/{args.augment_name}.json", 'r') as f:
            augment = json.load(f)

    convs = list(hybrid['conversations'].values())
    queries = list(map(lambda x: list(map(lambda y: hybrid['qas'][y]['current_query'], x)), convs))
    responses = list(map(lambda x: list(map(lambda y: hybrid['qas'][y]['long_response_to_query'], x)), convs))
    retrieved = list(map(lambda x: list(map(lambda y: hybrid['qas'][y]['correct_next_cands_ids'][0], x)), convs))
    retrieved_inputs = list(map(lambda x: list(map(lambda y: hybrid['all_candidates'][y]['linearized_input'], x)), retrieved))
    retrieved_keys = list(map(lambda x: list(map(lambda y: hybrid['all_candidates'][y]['table_key'], x)), retrieved))
    retrieved_keys_unique = list(map(lambda x: list(set(x)), retrieved_keys))
    retrieved_keys_unique = list(map(lambda x: list(filter(lambda y: y is not None, x)), retrieved_keys_unique))
    retrieved_keys_unique = list(map(lambda x: x[0], retrieved_keys_unique))
    tables_cands = list(map(lambda x: tables[x], retrieved_keys_unique))
    tables_flat = list(map(flatten_table_hybrid, tables_cands))
    output = []
    for x, i in enumerate(responses):
        tmp = []
        tmp.append("0: " + queries[x][0] + " 1: [MASK]")
        for j in range(1, len(i)):
            tmp.append(tmp[j-1][:-6] + responses[x][j-1] + " 0: " + queries[x][j] + " 1: [MASK]")
        output.append(tmp)



    # output_filtered = output_flat
    # retrieved_inputs_filtered = list(chain(*retrieved_inputs))
    for i, inputs in enumerate(retrieved_inputs):
            for j, src in enumerate(inputs):
                if src[1:3] == "TA" or src[1:3] == "RO" or src[1:3] == "CE":
                    inputs[j] = "[TABLE] " + tables_flat[i]
    # retrieved_inputs_filtered = list(filter(lambda x: x[1:3] == 'TA' or x[1:3] == 'PA', retrieved_inputs_flat))
    # output_filtered = [output_flat[i] for (i, x) in enumerate(retrieved_inputs_flat) if  x[1:3] == 'TA' or x[1:3] == 'PA']
    queries_filtered = list(chain(*queries))
    output_filtered = list(chain(*output))
    retrieved_inputs_filtered = list(chain(*retrieved_inputs))

    combined_lists = list(zip(output_filtered, retrieved_inputs_filtered))
    random.shuffle(combined_lists)
    output_filtered, retrieved_inputs_filtered = zip(*combined_lists)
    output_filtered = list(output_filtered)
    retrieved_inputs_filtered = list(retrieved_inputs_filtered)
    
    trim_size = 500
    augment_size = len(augment['context'])
    print(f"total data size without augmentation: {len(output_filtered)}")
    output_selected = output_filtered[:trim_size]
    retrieved_inputs_selected = retrieved_inputs_filtered[:trim_size]
    output_filtered_trn = output_selected[:len(output_selected)] + augment['context'][:int(0.8*augment_size)]
    retrieved_inputs_filtered_trn = retrieved_inputs_selected[:len(output_selected)] + augment['target'][:int(0.8*augment_size)]

    combined_lists = list(zip(output_filtered_trn, retrieved_inputs_filtered_trn))
    random.shuffle(combined_lists)
    output_filtered_trn, retrieved_inputs_filtered_trn = zip(*combined_lists)

    output_filtered_trn = list(output_filtered_trn)
    retrieved_inputs_filtered_trn = list(retrieved_inputs_filtered_trn)

    output_filtered_val = output_filtered[len(output_selected):len(output_selected)+trim_size]  + augment['context'][int(0.8*augment_size):]
    retrieved_inputs_filtered_val = retrieved_inputs_filtered[len(output_selected):len(output_selected)+trim_size] + augment['target'][int(0.8*augment_size):]

    combined_lists = list(zip(output_filtered_val, retrieved_inputs_filtered_val))
    random.shuffle(combined_lists)
    output_filtered_val, retrieved_inputs_filtered_val = zip(*combined_lists)

    output_filtered_val = list(output_filtered_val)
    retrieved_inputs_filtered_val = list(retrieved_inputs_filtered_val)

    print(f"training data size: {len(output_filtered_trn)}")
    print(f"augmented data size: {augment_size}" )
    print(f"validation data size: {len(output_filtered_val)}")

    data_dict_trn = {'context': output_filtered_trn, 'target': retrieved_inputs_filtered_trn}
    data_dict_val = {'context': output_filtered_val, 'target': retrieved_inputs_filtered_val}

    with open('data/retrieve.json', 'w') as f:
        json.dump(data_dict_trn, f)


    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
    tokenized_ds_trn = tokenize(data_dict_trn)
    ds_trn = CustomDataset(tokenized_ds_trn)
    tokenized_ds_val = tokenize(data_dict_val)
    ds_val = CustomDataset(tokenized_ds_val)

    batch_size = 256

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
    early_stop_cnt = 0
    for epoch in range(1, 1001):
        if early_stop_cnt > 5:
            break
            print("Early Stopped")
        
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
        print(f"epoch: {epoch}, loss:{loss_val:.4f}, R@1: {recall_top1:.3f}, R@5: {recall_top5:.3f}, R@10: {recall_top10:.3f}")



        if loss_val < best_loss:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_q.state_dict(),
                    'optimizer_state_dict': optim_q.state_dict(),
                    'val_loss': best_loss,
                    },
                    os.path.join(run_directory, f"model_q.pt")
                    
                )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_t.state_dict(),
                    'optimizer_state_dict': optim_t.state_dict(),
                    'val_loss': best_loss,
                    },
                    os.path.join(run_directory, f"model_t.pt")
                )
        else:
            early_stop_cnt += 1
        
            
    
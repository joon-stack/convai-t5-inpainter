import json

from itertools import chain

from transformers import AdamW, BertTokenizer, AutoTokenizer, AutoModel
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

def flatten_table(x):
    title = x['title']
    header = " ; ".join(x['header'])
    data = list(map(lambda x: " ; ".join(x), x['data']))
    data = " / ".join(data)
    return title + " / " + header + " / " + data

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



def process_dataset_gpt_for_retrieval():
    with open("data/gpt/new_json.json", 'r') as f:
        gpt = json.load(f)
    qas = list(map(lambda x: x['qas'], gpt))
    texts = list(map(lambda x: x['text'], gpt))
    tables = list(map(lambda x: x['table'], gpt))
    tables_flat = list(map(flatten_table, tables))
    questions = list(map(lambda y: list(map(lambda x: x['question'], y)), qas))
    answers = list(map(lambda y: list(map(lambda x: x['answer'], y)), qas))
    srcs = list(map(lambda y: list(map(lambda x: x['src'], y)), qas))

    context = []
    target = []
    for i, x in enumerate(questions):
        tmp = []
        tmp.append("0: " + questions[i][0] + " 1: [MASK]")
        for j in range(1, len(x)):
            tmp.append(tmp[j-1][:-6] + answers[i][j-1] + " 0: " + questions[i][j] + " 1: [MASK]")
        context.append(tmp)
        for src in srcs[i]:
            if src == 'table':
                target.append('[TABLE] ' + tables_flat[i])
            elif src == 'text':
                target.append('[PARAGRAPH] ' + texts[i])
                
    context = list(chain(*context))

    dataset = {'context': context, 'target': target}

    return dataset



    

# Checkpoint directory




if __name__ == "__main__":
    process_dataset_gpt_for_retrieval()
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment_name", type=str, default=None, help="the name of augmented dataset, None: no augment")
    parser.add_argument("--mode", type=str, default='add', help='the type of experiment, add: add augmented dataset to hybridialogue, solo_hy: use only hybrid, solo_gpt: use only gpt-aug, solo_inp: use only inpainter-aug')
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

    # load augmented data
    if args.mode == 'add':
        if args.augment_name:
            if args.augment_name == 'gpt':
                augment = process_dataset_gpt_for_retrieval()
            else:
                with open(f"data/retrieval/{args.augment_name}.json", 'r') as f:
                    augment = json.load(f)
        else:
            augment = {'context':[], 'target':[]}
    elif args.mode == 'solo_hy':
        augment = {'context':[], 'target':[]}
    elif args.mode == 'solo_gpt':
        augment = process_dataset_gpt_for_retrieval()
    elif args.mode == 'solo_inp':
        with open(f"data/retrieval/{args.augment_name}.json", 'r') as f:
            augment = json.load(f)


    
        
    with open("data/retrieval/aug_qot.pt.json", 'r') as f:
        augment_test = json.load(f)
    # load hybridialogue data
    with open("data/hybrid/experimental_data.json", 'r') as f:
        hybrid = json.load(f)

    with open("data/hybrid/traindev_tables.json", 'r') as f:
        tables = json.load(f)

    
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
    

    trim_size = len(augment_test['context'])
    augment_size = 0 if (args.augment_name == None and args.mode == 'add') or args.mode == 'solo_hy' else trim_size
    print(f"total data size without augmentation: {len(output_filtered)}")
    output_selected = output_filtered[:trim_size]
    retrieved_inputs_selected = retrieved_inputs_filtered[:trim_size]

    if augment_size > 0:
        aug_context = augment['context']
        aug_target = augment['target']

        combined_lists = list(zip(aug_context, aug_target))
        random.shuffle(combined_lists)
        aug_context, aug_target = zip(*combined_lists)
        aug_context = list(aug_context)
        aug_target = list(aug_target)

    aug_context_trim = aug_context[:trim_size] if augment_size > 0 else []
    aug_target_trim = aug_target[:trim_size] if augment_size > 0 else []

    if args.mode == 'add':

        output_filtered_trn = output_selected[:len(output_selected)] + aug_context_trim
        retrieved_inputs_filtered_trn = retrieved_inputs_selected[:len(output_selected)] + aug_target_trim

    elif args.mode == 'solo_hy':

        output_filtered_trn = output_selected[:len(output_selected)]
        retrieved_inputs_filtered_trn = retrieved_inputs_selected[:len(output_selected)]

    elif args.mode == 'solo_gpt' or 'solo_inp':
        output_filtered_trn = aug_context_trim
        retrieved_inputs_filtered_trn = aug_target_trim


    combined_lists = list(zip(output_filtered_trn, retrieved_inputs_filtered_trn))
    random.shuffle(combined_lists)
    output_filtered_trn, retrieved_inputs_filtered_trn = zip(*combined_lists)

    output_filtered_trn = list(output_filtered_trn)
    retrieved_inputs_filtered_trn = list(retrieved_inputs_filtered_trn)

    output_filtered_val = output_filtered[len(output_selected):len(output_selected)+trim_size] 
    retrieved_inputs_filtered_val = retrieved_inputs_filtered[len(output_selected):len(output_selected)+trim_size] 

    combined_lists = list(zip(output_filtered_val, retrieved_inputs_filtered_val))
    random.shuffle(combined_lists)
    output_filtered_val, retrieved_inputs_filtered_val = zip(*combined_lists)

    output_filtered_val = list(output_filtered_val)
    retrieved_inputs_filtered_val = list(retrieved_inputs_filtered_val)

    print(f"augmented data size: {augment_size}" )
    print(f"training data size: {len(output_filtered_trn)}")
    
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
    val_dataloader = DataLoader(ds_val, batch_size=len(ds_val))


    model_q = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    model_t = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    optim_q = AdamW(model_q.parameters(), lr=1e-4)
    optim_t = AdamW(model_t.parameters(), lr=1e-4)

    # for multiple GPUs
    # model_q = nn.DataParallel(model_q, device_ids = [0, 1, 2, 3])
    # model_t = nn.DataParallel(model_t, device_ids = [0, 1, 2, 3])
    # model_q.to(f'cuda:{model_q.device_ids[0]}')
    # model_t.to(f'cuda:{model_t.device_ids[0]}')

    model_q.to('cuda:0')
    model_t.to('cuda:0')

    best_loss = np.inf
    criterion = nn.CrossEntropyLoss()
    early_stop_cnt = 0

    for epoch in range(1, 501):
        if early_stop_cnt > 500:
            break
            print("Early Stopped")
        
        loss_batch = 0
        loss_val = 0
        
        model_q.train()
        model_t.train()

        
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

        model_q.eval()
        model_t.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids_q, attention_mask_q,  input_ids_t, attention_mask_t = batch
                input_ids_q = input_ids_q.to('cuda:0')
                attention_mask_q = attention_mask_q.to('cuda:0')
                input_ids_t = input_ids_t.to('cuda:0')
                attention_mask_t = attention_mask_t.to('cuda:0')

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
            if epoch % 100 == 0:
                print(f"epoch: {epoch}, loss:{loss_val:.4f}, R@1: {recall_top1:.3f}, R@5: {recall_top5:.3f}, R@10: {recall_top10:.3f}")



            if loss_val < best_loss:
                best_loss = loss_val
                early_stop_cnt = 0
                # print("Checkpoint saved")
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
        
            
    
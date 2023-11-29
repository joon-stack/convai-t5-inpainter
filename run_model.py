import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as Dataset_torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
import argparse
import random
import os
import json
from itertools import chain


seed = 2023

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print('pwd', os.getcwd())


def add_prefix(x):
    idx, item = x
    return str(idx % 2) + ": " + item

def preprocess(x):
    a = dict()
    a['text'] = list(map(lambda y: y['text'], x['utterances']))
    a['text'] = list(map(add_prefix, enumerate(a['text'])))
    return a

def process_dataset(dataset):
    processed_dataset = {}
    for sp in ['train', "validation", "test"]:

        dataset = taskmaster_dataset[sp]
        new_column = [None] * len(dataset)
        dataset = taskmaster_dataset[sp].add_column("text", new_column)
        updated_dataset = dataset.map(preprocess)
        merged_dataset = {}
        
        i = 0
        for dialog in updated_dataset['text']:
            if dialog != None:
                merged_dataset[i] = dialog
                i += 1
        masked_dataset = {'context':[], 'target':[]}
        for i, key in enumerate(merged_dataset):
            tmp = []
            dialog = merged_dataset[key]
            assert type(dialog) == type([])
            dialog_length = len(dialog)
            masked_idx = np.random.randint(0, dialog_length)
            masked_dataset['target'].append(dialog[masked_idx][3:] +" ")
            dialog[masked_idx] = dialog[masked_idx][:3] + "<MASK>"
            masked_dataset['context'].append(" ".join(dialog))

        processed_dataset[sp] = masked_dataset

    return processed_dataset

def process_dataset_qrecc(input_dataset):
    processed_dataset = {}
    for sp in ['train', 'test']:
        dataset = input_dataset[sp]
        conv_nos = list(map(lambda x: x['Conversation_no'], dataset))
        conv_last_nos_idx = []
        for i, x in enumerate(conv_nos):
            if i == len(conv_nos) - 1:
                break
            if x < conv_nos[i + 1]:
                conv_last_nos_idx.append(i)
        dialogs = list(map(lambda i: dataset[i]['Context'] + [dataset[i]['Question']] + [dataset[i]['Answer']], conv_last_nos_idx))
        dialogs_numbered = list(map(lambda x: list(map(lambda y: add_prefix(y), enumerate(x))), dialogs))
        masked_dataset = {'target':[], 'context':[]}
        for dialog in dialogs_numbered:
            dialog_length = len(dialog)
            masked_idx = np.random.randint(0, dialog_length)
            masked_dataset['target'].append(dialog[masked_idx][3:])
            dialog[masked_idx] = dialog[masked_idx][:3] + "<MASK>"
            masked_dataset['context'].append(" ".join(dialog))
        
        if sp == 'test':
            processed_dataset['validation'] = masked_dataset
        else: 
            processed_dataset[sp] = masked_dataset
    
    return processed_dataset

def process_dataset_orquac():
    processed_dataset = {}
    for sp in ['train', 'dev', 'test']:
        with open(f"data/{sp}.txt", 'r') as f:
            lines = f.readlines()
        dict_collection = [json.loads(line) for line in lines]
        conv_last_nos_idxs = []
        for i, col in enumerate(dict_collection):
            if col['qid'][-1] == '0':
                if i > 0:
                    conv_last_nos_idxs.append(i - 1)
        tmp = list(map(lambda i: (dict_collection[i]['history'] + [{'question': dict_collection[i]['question'], 'answer': dict_collection[i]['answer']}]), conv_last_nos_idxs))
        dialogs = list(map(lambda x: list(map(lambda y: ["0: " + y['question'], "1: " + y['answer']['text']], x)), tmp)) 
        dialogs_merged = list(map(lambda x: sum(x, []), dialogs))
        masked_dataset = {'target':[], 'context':[]}
        for dialog in dialogs_merged:
            dialog_length = len(dialog)
            masked_idx = np.random.randint(0, dialog_length)
            masked_dataset['target'].append(dialog[masked_idx][3:].replace("CANNOTANSWER", "I cannot answer the question."))
            dialog[masked_idx] = dialog[masked_idx][:3] + "<MASK>"
            masked_dataset['context'].append(" ".join(dialog).replace("CANNOTANSWER", "I cannot answer the question."))
        if sp != 'dev':
            processed_dataset[sp] = masked_dataset
        else:
            processed_dataset['validation'] = masked_dataset

    return processed_dataset

def flatten_table(x):
    title = x['title']
    header = " ".join(x['header'])
    data = list(map(lambda x: " ".join(x), x['data']))
    data = " ".join(data)
    return title + " " + header + " " + data

def process_dataset_gpt():
    with open("data/gpt/new_json.json", 'r') as f:
        gpt = json.load(f)
    qas = list(map(lambda x: x['qas'], gpt))
    texts = list(map(lambda x: x['text'], gpt))
    tables = list(map(lambda x: x['table'], gpt))
    tables_flat = list(map(flatten_table, tables))
    questions = list(map(lambda y: list(map(lambda x: x['question'], y)), qas))
    answers = list(map(lambda y: list(map(lambda x: x['answer'], y)), qas))
    srcs = list(map(lambda y: list(map(lambda x: x['src'], y)), qas))
    qa_pairs = list(map(lambda x: {"questions": questions[x], "answers": answers[x]}, list(range(len(questions)))))
    result = list(map(lambda x: list(chain.from_iterable(zip(x["questions"], x["answers"]))), qa_pairs))
    result_prefix = list(map(lambda x: list(map(add_prefix, enumerate(x))), result))

    data_train = result_prefix[:int(len(result_prefix) * 0.4)]
    data_val = result_prefix[int(len(result_prefix) * 0.4):len(result_prefix) // 2]
    data_test = result_prefix[len(result_prefix) // 2:]
    srcs_train = srcs[:int(len(result_prefix) * 0.4)]
    srcs_val = srcs[int(len(result_prefix) * 0.4):len(result_prefix) // 2]
    srcs_test = srcs[len(result_prefix) // 2:]
    tables_train = tables_flat[:int(len(result_prefix) * 0.4)]
    tables_val = tables_flat[int(len(result_prefix) * 0.4):len(result_prefix) // 2]
    tables_test = tables_flat[len(result_prefix) // 2:]
    texts_train = texts[:int(len(result_prefix) * 0.4)]
    texts_val = texts[int(len(result_prefix) * 0.4):len(result_prefix) // 2]
    texts_test = texts[len(result_prefix) // 2:]

    data = {'train': data_train, 'validation': data_val, 'test': data_test}
    srcs = {'train': srcs_train, 'validation': srcs_val, 'test': srcs_test}
    tables = {'train': tables_train, 'validation': tables_val, 'test': tables_test}
    texts = {'train': texts_train, 'validation': texts_val, 'test': texts_test}

    dataset = {}
    for sp in ['train', 'validation', 'test']:
        masked_dataset = {'context':[], 'target':[]}
        for i, dialog in enumerate(data[sp]):
            tmp = []
            assert type(dialog) == type([])
            dialog_length = len(dialog)
            masked_idx = np.random.randint(0, dialog_length)
            masked_dataset['target'].append(dialog[masked_idx][2:].strip())
            dialog[masked_idx] = dialog[masked_idx][:2] + " <MASK>"
            masked_dataset['context'].append("[DIALOG] " + " ".join(dialog) + " [SRC] " + srcs[sp][i][masked_idx // 2] + " [TABLE] " + tables[sp][i] + " [PARAGRAPH] " + texts[sp][i])
        dataset[sp] = masked_dataset

    return dataset

def tokenize(x):
    context_tokenized = tokenizer(x['context'],  padding=True, truncation=True, return_tensors='pt')
    target_tokenized = tokenizer(x['target'], padding=True, truncation=True, return_tensors='pt')
    return {
        "context": context_tokenized,
        "target": target_tokenized,
    }


    

class CustomDataset(Dataset_torch):
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
        
        # Perform any necessary transformations on the data here if needed
        # Return both the data and its corresponding key (or identifier)
        return context_input_ids, context_attention_mask, target_input_ids, target_attention_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='t', help="t for taskmaster (will be updated for qrecc and or-quac)")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decaying rate")
    parser.add_argument("--model_name", type=str, default="t5-small", help="the model name, default: t5-small")
    parser.add_argument("--epochs", type=int, default=100, help="the number of epochs")

    args = parser.parse_args()

    # Checkpoint directory
    os.mkdir("checkpoints") if not os.path.exists("checkpoints") else None
    # check GPU
    print('Cuda:', torch.cuda.is_available())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    # model = nn.DataParallel(model, device_ids = [0, 1, 2, 3])
    # model.to(f'cuda:{model.device_ids[0]}')
    model.to(device)

    print("Model on", device)


    my_special_tokens = {
        "additional_special_tokens": ["<MASK>"]
    }

    tokenizer.add_special_tokens(my_special_tokens)

    taskmaster_dataset = load_dataset("taskmaster1", "one_person_dialogs")
    qrecc_dataset = load_dataset("voidful/qrecc")

    if args.data == "t":
        dataset = process_dataset(taskmaster_dataset)

        ds = DatasetDict({
            'train': Dataset.from_dict(dataset['train']), 
            'validation': Dataset.from_dict(dataset['validation']),
            'test': Dataset.from_dict(dataset['test'])
        })

        tokenized_dataset = DatasetDict({
            'train': tokenize(ds['train']), 
            'validation': tokenize(ds['validation']),
            'test': tokenize(ds['test'])
        })
    elif args.data == "qt":
        dataset_tm = process_dataset(taskmaster_dataset)
        dataset_qrecc = process_dataset_qrecc(qrecc_dataset)
        dataset = {
                    'train': {'target':[], 'context':[]}, 
                    'validation': {'target':[], 'context':[]}, 
                    'test': {'target':[], 'context':[]}
                    }

        for sp in ['train', 'validation', 'test']:
            try:
                dataset[sp]['target'].extend(dataset_tm[sp]['target'])
                dataset[sp]['context'].extend(dataset_tm[sp]['context'])
                dataset[sp]['target'].extend(dataset_qrecc[sp]['target'])
                dataset[sp]['context'].extend(dataset_qrecc[sp]['context'])
            except:
                continue
        
        tokenized_dataset = DatasetDict({
            'train': tokenize(dataset['train']), 
            'validation': tokenize(dataset['validation']),
            'test': tokenize(dataset['test'])
        })
    
    elif args.data == "qot":
        dataset_tm = process_dataset(taskmaster_dataset)
        dataset_qrecc = process_dataset_qrecc(qrecc_dataset)
        dataset_orquac = process_dataset_orquac()
        dataset = {
                    'train': {'target':[], 'context':[]}, 
                    'validation': {'target':[], 'context':[]}, 
                    'test': {'target':[], 'context':[]}
                    }

        for sp in ['train', 'validation', 'test']:
            try:
                dataset[sp]['target'].extend(dataset_tm[sp]['target'])
                dataset[sp]['context'].extend(dataset_tm[sp]['context'])
                dataset[sp]['target'].extend(dataset_orquac[sp]['target'])
                dataset[sp]['context'].extend(dataset_orquac[sp]['context'])
                dataset[sp]['target'].extend(dataset_qrecc[sp]['target'])
                dataset[sp]['context'].extend(dataset_qrecc[sp]['context'])
                
            except:
                continue
        
        tokenized_dataset = DatasetDict({
            'train': tokenize(dataset['train']), 
            'validation': tokenize(dataset['validation']),
            'test': tokenize(dataset['test'])
        })
    elif args.data == "gpt":
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)
        my_special_tokens = {
        "additional_special_tokens": ["<MASK>", "[DIALOG]", "[SRC]", "[TABLE]", "[PARAGRAPH]"] 
        }

        tokenizer.add_special_tokens(my_special_tokens)
        dataset = process_dataset_gpt()
        tokenized_dataset = DatasetDict({
            'train': tokenize(dataset['train']), 
            'validation': tokenize(dataset['validation']),
            'test': tokenize(dataset['test'])
        })

    
    print(tokenized_dataset)



    dataset_train = CustomDataset(tokenized_dataset['train'])
    dataset_validation = CustomDataset(tokenized_dataset['validation'])
    dataset_test = CustomDataset(tokenized_dataset['test'])


    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size)
    val_dataloader = DataLoader(dataset_validation, batch_size=args.batch_size)

    num_epochs = args.epochs
    num_training_steps = args.epochs * len(train_dataloader)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_val_loss = float("inf")
    early_stop_cnt = 0
    for epoch in range(num_epochs):
        if early_stop_cnt > 4:
            break
        # training
        loss = 0
        model.train()
        for batch_i, batch in enumerate(train_dataloader):
            input_ids, attention_mask, labels, labels_attention_mask = batch
            labels_attention_mask = labels_attention_mask.type(torch.bool)
            labels_masked = torch.masked_fill(labels, ~labels_attention_mask, -100)
            # for multiple GPUs
            # input_ids = input_ids.to(f'cuda:{model.device_ids[0]}')
            # attention_mask = attention_mask.to(f'cuda:{model.device_ids[0]}')
            # labels_masked = labels_masked.to(f'cuda:{model.device_ids[0]}')
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_masked = labels_masked.to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_masked)
            optimizer.zero_grad()
            output.loss.mean().backward()
            optimizer.step()
            lr_scheduler.step()
            loss += output.loss.mean().item()
        avg_trn_loss = loss / len(train_dataloader)
        print(f"Train loss: {avg_trn_loss}")

        # validation
        loss = 0
        model.eval()
        for batch_i, batch in enumerate(val_dataloader):
            input_ids, attention_mask, labels, labels_attention_mask = batch
            labels_attention_mask = labels_attention_mask.type(torch.bool)
            labels_masked = torch.masked_fill(labels, ~labels_attention_mask, -100)
            # for multiple GPUs
            # input_ids = input_ids.to(f'cuda:{model.device_ids[0]}')
            # attention_mask = attention_mask.to(f'cuda:{model.device_ids[0]}')
            # labels_masked = labels_masked.to(f'cuda:{model.device_ids[0]}')
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_masked = labels_masked.to(device)
            
            with torch.no_grad():
                output = model(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                labels=labels_masked
                                )
            loss += output.loss.mean().item()

        avg_val_loss = loss / len(val_dataloader)
        print(f"Validation loss: {avg_val_loss}")
        if avg_val_loss < best_val_loss:
            print("Saving checkpoint!")
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                },
                f"checkpoints/epoch_{epoch}.pt"
            )
        else:
            early_stop_cnt += 1
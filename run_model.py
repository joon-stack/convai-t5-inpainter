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
    model = nn.DataParallel(model)
    # model.to(device)

    print("Model on", device)

    sep = tokenizer.eos_token

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
    for epoch in tqdm(range(num_epochs)):
        # training
        loss = 0
        model.train()
        for batch_i, batch in tqdm(enumerate(train_dataloader)):
            input_ids, attention_mask, labels, labels_attention_mask = batch
            labels_attention_mask = labels_attention_mask.type(torch.bool)
            labels_masked = torch.masked_fill(labels, ~labels_attention_mask, -100)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_masked = labels_masked.to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_masked)
            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()
            lr_scheduler.step()
            loss += output.loss.item()
        avg_trn_loss = loss / len(train_dataloader)
        print(f"Train loss: {avg_trn_loss}")

        # validation
        loss = 0
        model.eval()
        for batch_i, batch in tqdm(enumerate(val_dataloader)):
            input_ids, attention_mask, labels, labels_attention_mask = batch
            labels_attention_mask = labels_attention_mask.type(torch.bool)
            labels_masked = torch.masked_fill(labels, ~labels_attention_mask, -100)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_masked = labels_masked.to(device)
            
            with torch.no_grad():
                output = model(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                labels=labels_masked
                                )
            loss += output.loss.item()

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
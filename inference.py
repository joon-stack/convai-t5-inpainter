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
import re

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
    parser.add_argument("--checkpoint_name", type=str, help="checkpoint name, if this is None, the t5-small checkpoint is utilized for inference")
    parser.add_argument("--model_name", type=str, default="t5-small", help="model name (default: t5-small)")
    

    args = parser.parse_args()

    os.mkdir("results") if not os.path.exists("results") else None
    # check GPU
    print('Cuda:', torch.cuda.is_available())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    
    model = nn.DataParallel(model, device_ids = [0, 1, 2, 3])
    if args.checkpoint_name != None:
        model.load_state_dict(torch.load("checkpoints/"+args.checkpoint_name)['model_state_dict'])
        print("Model load success from ", "checkpoints/"+args.checkpoint_name)
    
    model.to(f'cuda:{model.device_ids[0]}')

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
        'test': Dataset.from_dict(dataset['test'])
    })

    tokenized_dataset = DatasetDict({
        'test': tokenize(ds['test'])
    })

    dataset_test = CustomDataset(tokenized_dataset['test'])

    test_dataloader = DataLoader(dataset_test, batch_size=64)

    # test
    fname = f"results/test_{args.checkpoint_name}.txt"
    with open(fname, 'w') as f:
        model.eval()
        for batch_i, batch in enumerate(test_dataloader):
            input_ids, attention_mask, labels, labels_attention_mask = batch
            labels_attention_mask = labels_attention_mask.type(torch.bool)
            labels_masked = torch.masked_fill(labels, ~labels_attention_mask, -100)
            input_ids = input_ids.to(f'cuda:{model.device_ids[0]}')
            attention_mask = attention_mask.to(f'cuda:{model.device_ids[0]}')
            labels_masked = labels_masked.to(f'cuda:{model.device_ids[0]}')
            outputs = model.module.generate(input_ids, max_new_tokens=100)
            decoded_context = list(map(lambda x: tokenizer.decode(x, skip_special_tokens=True), input_ids))
            decoded_pred = list(map(lambda x: tokenizer.decode(x, skip_special_tokens=True), outputs))
            decoded_label = list(map(lambda x: tokenizer.decode(x, skip_special_tokens=True), labels))
            for i in range(len(decoded_pred)):
                decoded_context_lines = re.split(r'[01]:', decoded_context[i])
                f.write("Context: " + "\n")
                cnt = 0
                for line in decoded_context_lines:
                    if line != "":
                        f.write(str(cnt%2)+ ":" + line + "\n")
                        cnt += 1

                f.write("Pred: " + decoded_pred[i] + "\n")
                f.write("Label: " + decoded_label[i] + "\n")
                f.write("--------------------------------\n")
        

        
        
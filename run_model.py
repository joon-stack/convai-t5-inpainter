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
import re
from collections import OrderedDict
import copy
from datetime import datetime
import gc

seed = 2023

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print('pwd', os.getcwd())

SPLIT_RATIO = [0.4, 0.1, 0.5]

def generate_datetime_key():
    """
    Generate a key based on the current date and time.
    """
    now = datetime.now()
    return now.strftime("%m%d%H%M%S")

def create_run_directory(base_directory, key):
    """
    Create a directory based on the random key under the specified base directory.
    """
    run_directory = os.path.join(base_directory, key)
    os.mkdir(run_directory) if not os.path.exists(run_directory) else None
    return run_directory


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
            dialog[masked_idx] = dialog[masked_idx][:3] + "[MASK]"
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
            dialog[masked_idx] = dialog[masked_idx][:3] + "[MASK]"
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
            dialog[masked_idx] = dialog[masked_idx][:3] + "[MASK]"
            masked_dataset['context'].append(" ".join(dialog).replace("CANNOTANSWER", "I cannot answer the question."))
        if sp != 'dev':
            processed_dataset[sp] = masked_dataset
        else:
            processed_dataset['validation'] = masked_dataset

    return processed_dataset

# for gpt
def process_dataset_orquac_2():
    processed_dataset = {}
    for sp in ['train', 'dev', 'test']:
        with open(f"data/{sp}.txt", 'r') as f:
            lines = f.readlines()
        dict_collection = [json.loads(line) for line in lines]
        conv_last_nos_idxs = []
        texts = list(map(lambda x: x['evidences'], dict_collection))
        for i, col in enumerate(dict_collection):
            if col['qid'][-1] == '0':
                if i > 0:
                    conv_last_nos_idxs.append(i - 1)
        conv_last_nos_idxs_fromzero = [-1]
        conv_last_nos_idxs_fromzero.extend(conv_last_nos_idxs)
        # print(conv_last_nos_idxs_fromzero)
        tmp = list(map(lambda i: (dict_collection[i]['history'] + [{'question': dict_collection[i]['question'], 'answer': dict_collection[i]['answer']}]), conv_last_nos_idxs))
        dialogs = list(map(lambda x: list(map(lambda y: ["0: " + y['question'], "1: " + y['answer']['text']], x)), tmp)) 
        dialogs_merged = list(map(lambda x: sum(x, []), dialogs))
        masked_dataset = {'target':[], 'context':[]}
        # print(len(texts), len(conv_last_nos_idxs_fromzero), len(dialogs_merged))
        for i, dialog in enumerate(dialogs_merged):
            dialog_length = len(dialog)
            masked_idx = np.random.randint(0, dialog_length)
            
            # print(conv_last_nos_idxs_fromzero[i] + 1 +masked_idx // 2)
            # print(len(texts), len(conv_last_nos_idxs_fromzero), len(dialogs_merged))
            # print(texts[conv_last_nos_idxs_fromzero[i] + 1 +masked_idx // 2][0])
            try:
                masked_dataset['target'].append(dialog[masked_idx][3:].replace("CANNOTANSWER", "I cannot answer the question."))
                dialog[masked_idx] = dialog[masked_idx][:3] + "[MASK]"
                masked_dataset['context'].append("[DIALOG] " + " ".join(dialog).replace("CANNOTANSWER", "I cannot answer the question.") + " [SRC] text [TABLE] [PARAGRAPH] " + " ".join(texts[conv_last_nos_idxs_fromzero[i] + 1 +masked_idx // 2]))
            except:
                pass

        if sp != 'dev':
            processed_dataset[sp] = masked_dataset
        else:
            processed_dataset['validation'] = masked_dataset

    return processed_dataset

def process_dataset_hybrid():
    with open("data/hybrid/experimental_data.json", 'r') as f:
        hybrid = json.load(f)
    with open("data/hybrid/traindev_tables.json", 'r') as f:
        tables = json.load(f)
    dataset = {}
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
    dialogs_merged = list(map(lambda i: list(chain.from_iterable(zip(queries[i], responses[i]))), range(len(responses))))
    dialogs_merged = list(map(lambda y: list(map(lambda x: str(x[0] % 2) + ": " + x[1], enumerate(y))), dialogs_merged))     

    for i, inputs in enumerate(retrieved_inputs):
        for j, src in enumerate(inputs):
            if src[1:3] == "TA" or src[1:3] == "RO" or src[1:3] == "CE":
                inputs[j] = "[TABLE] " + tables_flat[i]
    
    ratio = [0.8, 0.2, 0.0]


    combined_lists = list(zip(dialogs_merged, retrieved_inputs))
    random.shuffle(combined_lists)
    dialogs_merged, retrieved_inputs = zip(*combined_lists)
    size = len(dialogs_merged)
    test_dialog =  dialogs_merged[int((ratio[0] + ratio[1]) * size):] if ratio[2] == 0.0 else None
    test_inputs =  retrieved_inputs[int((ratio[0] + ratio[1]) * size):] if ratio[2] == 0.0 else None
    dialog_dict = {
                    'train': dialogs_merged[:int(ratio[0] * size)],
                    'validation': dialogs_merged[int(ratio[0] * size):int((ratio[0] + ratio[1]) * size)],
                    'test': dialogs_merged[int((ratio[0] + ratio[1]) * size):],
                    }
    inputs_dict = {
                    'train': retrieved_inputs[:int(ratio[0] * size)],
                    'validation': retrieved_inputs[int(ratio[0] * size):int((ratio[0] + ratio[1]) * size)],
                    'test': retrieved_inputs[int((ratio[0] + ratio[1]) * size):],
                    }
                

    for sp in ['train', 'validation', 'test']:
        masked_dataset = {'context':[], 'target':[]}
        try:
            for i, dialog in enumerate(dialog_dict[sp]):
                dialog_length = len(dialog)
                if args.mask == 'random':
                # for masked_idx in range(dialog_length):
                    masked_idx = np.random.randint(0, dialog_length)
                    masked_dataset['target'].append(dialog[masked_idx][3:])
                    src = inputs_dict[sp][i][masked_idx // 2]
                    dialog_copy = copy.copy(dialog)
                    dialog_copy[masked_idx] = dialog[masked_idx][:3] + "[MASK]"
                    masked_dataset['context'].append("[DIALOG] " + " ".join(dialog_copy)+ src.replace("$", "") )
                elif args.mask == 'all':
                    for masked_idx in range(dialog_length):
                        masked_idx = np.random.randint(0, dialog_length)
                        masked_dataset['target'].append(dialog[masked_idx][3:])
                        src = inputs_dict[sp][i][masked_idx // 2]
                        dialog_copy = copy.copy(dialog)
                        dialog_copy[masked_idx] = dialog[masked_idx][:3] + "[MASK]"
                        masked_dataset['context'].append("[DIALOG] " + " ".join(dialog_copy) + src.replace("$", "") )
                dataset[sp] = masked_dataset
        except:
            continue
    return dataset



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
    header_flat = " ; ".join(header_flat)
    data = x['data']
    data_flat = list(map(lambda z: list(map(lambda y: " ".join(y[:-1]), z)), data))
    data_flat = list(map(lambda x: " ; ".join(x), data_flat))
    data_flat = " / ".join(data_flat)
    result = title + " / " + header_flat + " / " + data_flat
    return result


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
    parser.add_argument("--data", type=str, default='t', help="t: taskmaster, qt: qrecc + taskmaster, qot: qrecc + orquac + taskmaster, gpt: gpt-made, hybrid: hybridialgoue, hybridgpt: hybrid + gpt")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decaying rate")
    parser.add_argument("--model_name", type=str, default="t5-small", help="the model name, default: t5-small")
    parser.add_argument("--epochs", type=int, default=100, help="the number of epochs")
    parser.add_argument("--mode", type=str, default='train', help='whether to train or augment (inference), or augment_test')
    parser.add_argument("--checkpoint_name", type=str, help="checkpoint name, if this is None, the t5-small checkpoint is utilized for inference")
    parser.add_argument("--mask", type=str, default='random', help="hybrid masking option, random: randomly mask one q or a, all: mask all")
    parser.add_argument("--mask_gpt", type=str, default='random', help="gpt masking option, random: randomly mask one q or a, all: mask all")
    parser.add_argument("--early_stop", type=int, default=5, help="epochs to early stop")
    parser.add_argument("--aug_mode", type=str, default='retrieve', help="purpose of augmentation. retrieve: for retrieval (no details), eval: for evaluation (with details)")
    args = parser.parse_args()

    
    torch.cuda.empty_cache()
    gc.collect()

    # Checkpoint directory
    if args.mode == 'train':
        os.mkdir("checkpoints") if not os.path.exists("checkpoints") else None
        base_directory = "checkpoints"
        random_key = generate_datetime_key()
        run_directory = create_run_directory(base_directory, random_key)
        print(f"Random Key: {random_key}")
        print(f"Run Directory: {run_directory}")
    # check GPU
    print('Cuda:', torch.cuda.is_available())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    my_special_tokens = {
    "additional_special_tokens": ["[MASK]", "[DIALOG]", "[TABLE]", "[PARAGRAPH]", ';', '/'] 
    }
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    # model = nn.DataParallel(model, device_ids = [0, 1, 2, 3])
    # model.to(f'cuda:{model.device_ids[0]}')
    if args.checkpoint_name != None:
        state_dict = torch.load("checkpoints/"+args.checkpoint_name)['model_state_dict']
        keys = state_dict.keys()
        values = state_dict.values()
        new_keys = []
        for key in keys:
            new_key = key.replace("module.", "")
            new_keys.append(new_key)
        new_dict = OrderedDict(list(zip(new_keys, values)))
        model.load_state_dict(new_dict)
        # model.load_state_dict(torch.load("checkpoints/"+args.checkpoint_name)['model_state_dict'])
        print("Model load success from ", "checkpoints/"+args.checkpoint_name)
    
    
    model.to(device)

    print("Model on", device)



    tokenizer.add_special_tokens(my_special_tokens)


    

    if args.data == "t":
        taskmaster_dataset = load_dataset("taskmaster1", "one_person_dialogs")
        qrecc_dataset = load_dataset("voidful/qrecc")
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
        taskmaster_dataset = load_dataset("taskmaster1", "one_person_dialogs")
        qrecc_dataset = load_dataset("voidful/qrecc")
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
        taskmaster_dataset = load_dataset("taskmaster1", "one_person_dialogs")
        qrecc_dataset = load_dataset("voidful/qrecc")
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

        combined_lists = list(zip(result_prefix, srcs, tables_flat, texts))
        random.shuffle(combined_lists)
        result_prefix, srcs, tables_flat, texts = zip(*combined_lists)


        size = len(result_prefix)
        data_train = result_prefix[:int(SPLIT_RATIO[0]*size)]
        data_val = result_prefix[int(SPLIT_RATIO[0]*size):int(SPLIT_RATIO[1]*size + SPLIT_RATIO[0]*size)]
        data_test = result_prefix[int(SPLIT_RATIO[1]*size + SPLIT_RATIO[0]*size):]
        srcs_train = srcs[:int(SPLIT_RATIO[0]*size)]
        srcs_val = srcs[int(SPLIT_RATIO[0]*size):int(SPLIT_RATIO[1]*size + SPLIT_RATIO[0]*size)]
        srcs_test = srcs[int(SPLIT_RATIO[1]*size + SPLIT_RATIO[0]*size):]
        tables_train = tables_flat[:int(SPLIT_RATIO[0]*size)]
        tables_val = tables_flat[int(SPLIT_RATIO[0]*size):int(SPLIT_RATIO[1]*size + SPLIT_RATIO[0]*size)]
        tables_test = tables_flat[int(SPLIT_RATIO[1]*size + SPLIT_RATIO[0]*size):]
        texts_train = texts[:int(SPLIT_RATIO[0]*size)]
        texts_val = texts[int(SPLIT_RATIO[0]*size):int(SPLIT_RATIO[1]*size + SPLIT_RATIO[0]*size)]
        texts_test = texts[int(SPLIT_RATIO[1]*size + SPLIT_RATIO[0]*size):]
        print(data_test[0])

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
                if args.mask_gpt == 'all':
                    for masked_idx in range(dialog_length):
                        masked_dataset['target'].append(dialog[masked_idx][2:].strip())
                        dialog_copy = copy.copy(dialog)
                        dialog_copy[masked_idx] = dialog[masked_idx][:2] + " [MASK]"
                        source = srcs[sp][i][masked_idx // 2]
                        if source == 'text':
                            processed_tmp = "[DIALOG] " + " ".join(dialog_copy) + " [PARAGRAPH] " + texts[sp][i]
                            # processed_tmp = " [PARAGRAPH] " + texts[sp][i] + " [DIALOG] " + " ".join(dialog_copy) 
                        elif source == 'table':
                            processed_tmp = "[DIALOG] " + " ".join(dialog_copy) + " [TABLE] " + tables[sp][i] 
                            # processed_tmp = " [TABLE] " + tables[sp][i] + " [DIALOG] " + " ".join(dialog_copy)
                        masked_dataset['context'].append(processed_tmp)
                elif args.mask_gpt == 'random':
                    masked_idx = np.random.randint(0, dialog_length)
                    masked_dataset['target'].append(dialog[masked_idx][2:].strip())
                    dialog_copy = copy.copy(dialog)
                    dialog_copy[masked_idx] = dialog[masked_idx][:2] + " [MASK]"
                    source = srcs[sp][i][masked_idx // 2]
                    if source == 'text':
                        processed_tmp = "[DIALOG] " + " ".join(dialog_copy) + " [PARAGRAPH] " + texts[sp][i]
                    elif source == 'table':
                        processed_tmp = "[DIALOG] " + " ".join(dialog_copy) + " [TABLE] " + tables[sp][i] 
                    masked_dataset['context'].append(processed_tmp)
                dataset[sp] = masked_dataset


        with open('data/gpt_test.json', 'w') as f:
            json.dump(dataset, f)

        tokenized_dataset = DatasetDict({
            'train': tokenize(dataset['train']), 
            'validation': tokenize(dataset['validation']),
            'test': tokenize(dataset['test'])
        })

    elif args.data == "hybrid":

        dataset = process_dataset_hybrid()
        with open('data/hybrid_test.json', 'w') as f:
            json.dump(dataset['validation'], f)

        try:

            tokenized_dataset = DatasetDict({
                'train': tokenize(dataset['train']), 
                'validation': tokenize(dataset['validation']),
                # 'test': tokenize(dataset['test'])
            })
        except:
            tokenized_dataset = DatasetDict({
                'train': tokenize(dataset['train']), 
                'validation': tokenize(dataset['validation']),
                'test': tokenize(dataset['test'])
            })
    
    elif args.data == "hybridgpt":
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

        combined_lists = list(zip(result_prefix, srcs, tables_flat, texts))
        random.shuffle(combined_lists)
        result_prefix, srcs, tables_flat, texts = zip(*combined_lists)


        size = len(result_prefix)
        data_train = result_prefix[:int(SPLIT_RATIO[0]*size)]
        data_val = result_prefix[int(SPLIT_RATIO[0]*size):int(SPLIT_RATIO[1]*size + SPLIT_RATIO[0]*size)]
        data_test = result_prefix[int(SPLIT_RATIO[1]*size + SPLIT_RATIO[0]*size):]
        srcs_train = srcs[:int(SPLIT_RATIO[0]*size)]
        srcs_val = srcs[int(SPLIT_RATIO[0]*size):int(SPLIT_RATIO[1]*size + SPLIT_RATIO[0]*size)]
        srcs_test = srcs[int(SPLIT_RATIO[1]*size + SPLIT_RATIO[0]*size):]
        tables_train = tables_flat[:int(SPLIT_RATIO[0]*size)]
        tables_val = tables_flat[int(SPLIT_RATIO[0]*size):int(SPLIT_RATIO[1]*size + SPLIT_RATIO[0]*size)]
        tables_test = tables_flat[int(SPLIT_RATIO[1]*size + SPLIT_RATIO[0]*size):]
        texts_train = texts[:int(SPLIT_RATIO[0]*size)]
        texts_val = texts[int(SPLIT_RATIO[0]*size):int(SPLIT_RATIO[1]*size + SPLIT_RATIO[0]*size)]
        texts_test = texts[int(SPLIT_RATIO[1]*size + SPLIT_RATIO[0]*size):]
        print(data_test[0])

        data = {'train': data_train, 'validation': data_val, 'test': data_test}
        srcs = {'train': srcs_train, 'validation': srcs_val, 'test': srcs_test}
        tables = {'train': tables_train, 'validation': tables_val, 'test': tables_test}
        texts = {'train': texts_train, 'validation': texts_val, 'test': texts_test}

        dataset = {}
        for sp in ['train', 'validation', 'test']:
            masked_dataset = {'context':[], 'target':[]}
            try:
                for i, dialog in enumerate(data[sp]):
                    tmp = []
                    assert type(dialog) == type([])
                    dialog_length = len(dialog)
                    if args.mask_gpt == 'all':
                        for masked_idx in range(dialog_length):
                            masked_dataset['target'].append(dialog[masked_idx][2:].strip())
                            dialog_copy = copy.copy(dialog)
                            dialog_copy[masked_idx] = dialog[masked_idx][:2] + " [MASK]"
                            source = srcs[sp][i][masked_idx // 2]
                            if source == 'text':
                                processed_tmp = "[DIALOG] " + " ".join(dialog_copy) + " [PARAGRAPH] " + texts[sp][i]
                                # processed_tmp = " [PARAGRAPH] " + texts[sp][i] + " [DIALOG] " + " ".join(dialog_copy) 
                            elif source == 'table':
                                processed_tmp = "[DIALOG] " + " ".join(dialog_copy) + " [TABLE] " + tables[sp][i] 
                                # processed_tmp = " [TABLE] " + tables[sp][i] + " [DIALOG] " + " ".join(dialog_copy)
                            masked_dataset['context'].append(processed_tmp)
                    elif args.mask_gpt == 'random':
                        masked_idx = np.random.randint(0, dialog_length)
                        masked_dataset['target'].append(dialog[masked_idx][2:].strip())
                        dialog_copy = copy.copy(dialog)
                        dialog_copy[masked_idx] = dialog[masked_idx][:2] + " [MASK]"
                        source = srcs[sp][i][masked_idx // 2]
                        if source == 'text':
                            processed_tmp = "[DIALOG] " + " ".join(dialog_copy) + " [PARAGRAPH] " + texts[sp][i]
                        elif source == 'table':
                            processed_tmp = "[DIALOG] " + " ".join(dialog_copy) + " [TABLE] " + tables[sp][i] 
                        masked_dataset['context'].append(processed_tmp)
                    dataset[sp] = masked_dataset
            except:
                continue
        
        dataset_hy = process_dataset_hybrid()

        for sp in ['train', 'validation', 'test']:
            try:
                dataset[sp]['target'].extend(dataset_hy[sp]['target'])
                dataset[sp]['context'].extend(dataset_hy[sp]['context'])
                
            except:
                continue

        tokenized_dataset = DatasetDict({
            'train': tokenize(dataset['train']), 
            'validation': tokenize(dataset['validation']),
            'test': tokenize(dataset['test'])
        })

        with open('data/hybridgpt_test.json', 'w') as f:
            json.dump(dataset, f)


    dataset_train = CustomDataset(tokenized_dataset['train'])
    dataset_validation = CustomDataset(tokenized_dataset['validation'])
    

    print(f"Train dataset size: {len(dataset_train)}")
    print(f"Validation dataset size: {len(dataset_validation)}")

    try:
        dataset_test = CustomDataset(tokenized_dataset['test'])
        print(f"Test dataset size: {len(dataset_test)}")
    except:
        pass

    mode = args.mode

    if args.mode == 'augment_test':
        # paragraph = "[PARAGRAPH] Traffic is a 2000 American movie crime drama film directed by Steven Soderbergh and written by Stephen Gaghan . It explores the illegal drug trade from a number of perspectives : users , enforcers , politicians , and traffickers . Their stories are edited together throughout the film , although some of the characters do not meet each other . The film is an adaptation of the 1989 British Channel 4 television series Traffik . 20th Century Fox , the original financiers of the film , demanded that Harrison Ford play a leading role and that significant changes to the screenplay be made . Soderbergh refused and proposed the script to other major Hollywood studios , but it was rejected because of the three-hour running time and the subject matter - Traffic is more of a political film than most Hollywood productions . USA Films , however , liked the project from the start and offered the filmmakers more money than Fox . Soderbergh operated the camera himself and adopted a distinctive color grade for each storyline so that audiences could tell them apart . Traffic was critically acclaimed and earned numerous awards , including four Oscars : Best Director for Steven Soderbergh , Best Supporting Actor for Benicio del Toro , Best Adapted Screenplay for Stephen Gaghan and Best Film Editing for Stephen Mirrione . It was also a commercial success with a worldwide box-office revenue total of $ 207.5 million , well above its estimated $ 46 million budget . In 2004 , USA Network ran a miniseries - also called Traffic - based on this film and the 1989 British television series ."
        # data_test = ["0: [MASK] 1: Yes", "0: Who is the voice of the character who played the role of the 'Giant Mechanical Man' in the movie Traffic? 1: Topher Grace 0: What year did Topher Grace play the role of the 'Giant Mechanical Man' in the movie Traffic? 1: 2000 0: Who directed the movie Traffic? 1: Steven Soderbergh 0: [MASK] 1: Yes ", "[DIALOG] 0: [MASK] 1: Yes " + paragraph, "[DIALOG] 0: Who is the voice of the character who played the role of the 'Giant Mechanical Man' in the movie Traffic? 1: Topher Grace 0: What year did Topher Grace play the role of the 'Giant Mechanical Man' in the movie Traffic? 1: 2000 0: Who directed the movie Traffic? 1: Steven Soderbergh 0: [MASK] 1: Yes " + paragraph]
        paragraph = "[PARAGRAPH] Harvest is the fourth studio album by Canadian musician Neil Young , released in February 1972 on Reprise Records , catalogue MS 2032 . It featured the London Symphony Orchestra on two tracks and vocals by noted guests David Crosby , Graham Nash , Linda Ronstadt , Stephen Stills , and James Taylor . It topped the Billboard 200 album chart for two weeks , and spawned two hit singles , Old Man , which peaked at # 31 on the Billboard Hot 100 , and Heart of Gold , which reached No . 1 . It was the best-selling album of 1972 in the United States ."

        table = "[TABLE] List of best-selling albums by year in the United States / Year ; Performing artist ( s ) ; Nationality ; Album / 1970 ; Simon and Garfunkel ; United States ; Bridge over Troubled Water / 1971 ; Various Artists ; - ; Jesus Christ Superstar / 1972 ; Neil Young ; Canada ; Harvest / 1973 ; War ; United States ; The World Is a Ghetto / 1974 ; Elton John ; United Kingdom ; Goodbye Yellow Brick Road / 1975 ; Elton John ; United Kingdom ; Elton John 's Greatest Hits / 1976 ; Peter Frampton ; UK/USA ; Frampton Comes Alive / 1977 ; Fleetwood Mac ; UK/USA ; Rumours / 1978 ; Soundtrack/ Bee Gees ; - ; Saturday Night Fever / 1979 ; Billy Joel ; United States ; 52nd Street"

        
        dialog_1 = "0: [MASK] 1: Yes"
        dialog_2 = "0: Who is the best-selling album by year in the United States? 1: Neil Young 0: What country is Neil Young from? 1: Canada 0: What year did Neil Young release Harvest? 1: 1972 0: [MASK] 1: Yes"
        dialog_3 = "0: [MASK] 1: 1972"
        dialog_4 = "0: Who is the best-selling album by year in the United States? 1: Neil Young 0: What country is Neil Young from? 1: Canada 0: Is Harvest the fourth studio album by Canadian musician Neil Young? 1: Yes 0: [MASK] 1: 1972"

        print(type(dialog_1), type(dialog_2), type(dialog_3), type(dialog_4), type(paragraph), type(table))
        data_test = [dialog_1, dialog_2, "[DIALOG] " + dialog_1 + " " + paragraph, "[DIALOG] " + dialog_2 + " " + paragraph, dialog_3, dialog_4, "[DIALOG] " + dialog_3 + " " + table, "[DIALOG] " + dialog_4 + " " + table] 
    

    
    if mode == 'train':

        # dataset_train = CustomDataset(tokenized_dataset['train'])
        # dataset_validation = CustomDataset(tokenized_dataset['validation'])
        # dataset_test = CustomDataset(tokenized_dataset['test'])

        # print(f"Train dataset size: {len(dataset_train)}")
        # print(f"Validation dataset size: {len(dataset_validation)}")
        # print(f"Test dataset size: {len(dataset_test)}")


        train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size)
        val_dataloader = DataLoader(dataset_validation, batch_size=args.batch_size)

        num_epochs = args.epochs
        num_training_steps = args.epochs * len(train_dataloader)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        best_val_loss = float("inf")
        early_stop_cnt = 0
        for epoch in range(num_epochs):
            if early_stop_cnt > args.early_stop:
                print("EARLY STOPPED")
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
            print(f"Train loss: {avg_trn_loss:.4f}")

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
            print(f"Validation loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                early_stop_cnt = 0
                print("Saving checkpoint!")
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    },
                    os.path.join(run_directory, f"model.pt")
                )
            else:
                early_stop_cnt += 1

    elif mode == 'augment_test':
        model.to(device)
        model_base = T5ForConditionalGeneration.from_pretrained('t5-base')
        state_dict = torch.load("checkpoints/qot-base.pt")['model_state_dict']
        keys = state_dict.keys()
        values = state_dict.values()
        new_keys = []
        for key in keys:
            new_key = key.replace("module.", "")
            new_keys.append(new_key)
        new_dict = OrderedDict(list(zip(new_keys, values)))
        model_base.load_state_dict(new_dict)
        print("Model base load success from ", "checkpoints/qot-base.pt")
        model_base.to(device)
        model.eval()
        model_base.eval()
        with torch.no_grad():
            for n, data in enumerate(data_test):
                tok = tokenizer(data.strip(), truncation=True, return_tensors='pt')
                input_ids = tok['input_ids'].to(device)
                outputs = model.generate(input_ids, max_new_tokens=100)
                outputs_base = model_base.generate(input_ids, max_new_tokens=100)
                decoded_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)  
                decoded_pred_base =  tokenizer.decode(outputs_base[0], skip_special_tokens=True)    
                print(f"Ours: {decoded_pred}, Baseline: {decoded_pred_base}")


    elif mode == 'augment':

    
        print(data_test[0])
        print(f"Data size for augmentation: {len(data_test)}")
        print(len(tables_test), len(texts_test))

        fname = f"results/aug_{args.checkpoint_name}.txt" if args.aug_mode == 'eval' else f"data/retrieval/aug_{args.checkpoint_name}.json"
        
        model.to(device)
        
        model.eval()
        with torch.no_grad():

            retrieval_dataset = {'context': [], 'target': []}
            with open(fname, 'w') as f:
                for n, data in enumerate(data_test):
                    tmp = ""
                    sources = []
                    for i in range(len(data) // 2):
                        tmp = tmp + " " + data[2*i][:2] + " [MASK] "  + data[2*i+1] 
                        # print(tmp)
                        # tmp_add = tmp + " [SRC] " + srcs_test[n][i]
                        source = srcs_test[n][i]
                        if source == 'table':
                            tmp_add = tmp + " [TABLE] " + tables_test[n]
                        elif source == 'text':
                            tmp_add = tmp + " [PARAGRAPH] " + texts_test[n]
                        tok = tokenizer(tmp_add.strip(), truncation=True, return_tensors='pt')
                        input_ids = tok['input_ids'].to(device)
                        outputs = model.generate(input_ids, max_new_tokens=100)
                        decoded_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)     
                        tmp = tmp.replace("[MASK]", decoded_pred)
                        tmp = tmp.strip()
                        sources.append(source)
                        sources.append(source)
                    decoded_context_lines = re.split(r'[01]: ', tmp)[1:]

                    cnt = 0
                    if args.aug_mode == 'eval':
                        
                        f.write("PREDICTION\n")
                        for i, line in enumerate(decoded_context_lines):
                            if line != "":
                                try:
                                    f.write(sources[cnt] + " " + str(cnt%2)+ ":" + line + "\n")
                                    cnt += 1
                                except:
                                    f.write(str(cnt%2)+ ":" + line + "\n")
                        f.write("=============================" + "\n")
                        f.write("GROUND TRUTH\n")
                        cnt = 0
                        for line in data:
                            if line != "":
                                f.write(line + "\n")
                                cnt += 1
                        f.write("=============================" + "\n")
                    
                    elif args.aug_mode == 'retrieve':
                        dialog = []
                        chat_mask = ""
                        chat_full = ""
                        
                        for i, line in enumerate(decoded_context_lines):
                            if line != "":
                                tmp_mask = str(cnt%2)+ ": " + line.strip() + " " if cnt%2 == 0 else str(cnt%2) + ": [MASK]"
                                tmp_full = str(cnt%2)+ ": " + line.strip() + " "
                                chat_mask = chat_full + tmp_mask
                                chat_full += tmp_full
                                
                                if cnt%2 == 1:
                                    try:
                                    
                                    # print(f'i: {i}, n: {n}')
                                        retrieval_dataset['context'].append(chat_mask)
                                        src = "[TABLE] " + tables_test[n] if sources[i] == 'table' else "[PARAGRAPH] " + texts_test[n]
                                        retrieval_dataset['target'].append(src)
                                    except:
                                        pass

                                cnt += 1
                if args.aug_mode == 'retrieve':
                    json.dump(retrieval_dataset, f)


                
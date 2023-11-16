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
import copy
import json

seed = 2023

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print('pwd', os.getcwd())

def process_dataset_from_json(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
        candidates = list(map(lambda x: (x['qid'], x['candidates']), data))
        candidates = list(map(lambda x: (x[0], list(map(lambda x: "1: " + x, x[1]))), candidates))
        candidates_dict = dict(candidates)

    return candidates_dict

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
    elif args.data == "ottqa":
        candidates = process_dataset_from_json("data/new_dev_clean.json")


    # test
    fname = f"results/aug_test_{args.checkpoint_name}.txt"
    with open(fname, 'w') as f:
        model.eval()
        # tmp_cnt = 0
        for key in tqdm(candidates):
            # if tmp_cnt == 5:
            #     break
            contexts = candidates[key]
            tmp = ""
            for i, context in enumerate(contexts):
                tmp = tmp + " 0: <MASK> " + context
                print(tmp)
                context_tokenized = tokenizer(tmp.strip(), truncation=True, return_tensors='pt')
                input_ids = context_tokenized['input_ids'].to(device)
                outputs = model.module.generate(input_ids, max_new_tokens=100)
                decoded_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                tmp = tmp.replace("<MASK>", decoded_pred)
                print(tmp)
            tmp = tmp.strip()
            # print(tmp)
            decoded_context_lines = re.split(r'[01]:', tmp)
            # tmp_cnt += 1
            cnt = 0
            for line in decoded_context_lines:
                if line != "":
                    f.write(str(cnt%2)+ ":" + line + "\n")
                    cnt += 1
            f.write("=============================" + "\n")

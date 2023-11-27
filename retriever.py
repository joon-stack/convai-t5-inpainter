import json

from itertools import chain

from transformers import TapasConfig, TapasForQuestionAnswering, AdamW, TapasTokenizer, BertTokenizer
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import pandas as pd

def tokenize(x):
    context_tokenized = tokenizer(x['context'],  padding=True, truncation=True, return_tensors='pt')
    target_tokenized = tokenizer(x['target'], padding=True, truncation=True, return_tensors='pt')
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
    
class CustomTapas(nn.Module):
    def __init__(self):
        super(CustomTapas, self).__init__()
        self.tapas = TapasForQuestionAnswering.from_pretrained("google/tapas-tiny")
        self.proj = nn.Linear(128, 256)

    def forward(self, x):
        x = self.tapas(x)
        x = self.proj(x)
        return x


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
    tmp.append("0: " + queries[x][0] + " 1: <MASK>")
    for j in range(1, len(i)):
        tmp.append(tmp[j-1][:-6] + responses[x][j-1] + " 0: " + queries[x][j] + " 1: <MASK>")
    output.append(tmp)


queries_flat = list(chain(*queries))
output_flat = list(chain(*output))
retrieved_inputs_flat = list(chain(*retrieved_inputs))
retrieved_inputs_filtered = list(filter(lambda x: x[1:3] == 'TA' or x[1:3] == 'PA', retrieved_inputs_flat))
output_filtered = [output_flat[i] for (i, x) in enumerate(retrieved_inputs_flat) if  x[1:3] == 'TA' or x[1:3] == 'PA']
retrieved_inputs_filtered = list(map(lambda x: x.replace("$", "").strip(), retrieved_inputs_filtered))
data_dict = {'context': output_filtered, 'target': retrieved_inputs_filtered}

model_q = CustomTapas()
model_w = CustomTapas()

tokenizer = BertTokenizer.from_pretrained("google/tapas-tiny")
my_special_tokens = {
        "additional_special_tokens": ["<MASK>"]
    }

tokenizer.add_special_tokens(my_special_tokens)

tokenized_ds = tokenize(data_dict)

optim_q = AdamW(model_q.parameters(), lr=5e-5)
optim_t = AdamW(model_t.parameters(), lr=5e-5)

model_q.train()
model_t.train()
criterion = nn.CrossEntropyLoss()
for epoch in range(2):
    for batch in train_dataloader:
        input_ids_q = batch["q"]["input_ids"]
        attention_mask_q = batch["q"]["attention_mask"]
        token_type_ids_q = batch["q"]["token_type_ids"]
        labels_q = batch["q"]["labels"]

        input_ids_t = batch["t"]["input_ids"]
        attention_mask_t = batch["t"]["attention_mask"]
        token_type_ids_t = batch["t"]["token_type_ids"]
        labels_t = batch["t"]["labels"]

        optim_q.zero_grad()
        optim_t.zero_grad()


        # B*256 shape
        h_q = model(
            input_ids=input_ids_q,
            attention_mask=attention_mask_q,
            token_type_ids=token_type_ids_q
            labels=labels_q
        ).values
        # B*256 shape
        h_t = model(
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            token_type_ids=token_type_ids_t
            labels=labels_t
        ).values

        # B*B shape
        logits = h_q @ h_t.T
        labels = torch.eye(logits.shape[0])
        loss = criterion(logits, labels)
        loss.backward()
        optim_q.step()
        optim_t.step()

        
        
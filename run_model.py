from transformers import T5Tokenizer, T5Model
import torch

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5Model.from_pretrained("t5-small")

input_ids = tokenizer(
    "Studies have been shown that owning a dog is good for you", return_tensors="pt"
).input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

# check GPU
print('Cuda:', torch.cuda.is_available())

# forward pass
outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).cuda()
last_hidden_states = outputs.last_hidden_state

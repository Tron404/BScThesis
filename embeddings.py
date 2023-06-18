import time
import numpy as np
import torch

from tqdm import tqdm
from npy_append_array import NpyAppendArray

from transformers import BertModel, MT5EncoderModel, XLMRobertaModel, ErnieMModel
from transformers import BertTokenizer, T5Tokenizer, XLMRobertaTokenizer, ErnieMTokenizer 

device = torch.device("cuda:0")

def load_tokenizer_model(model_name):
  func_model = BertModel
  func_tokenizer = BertTokenizer

  if "mt5-base" == model_name:
    func_model = MT5EncoderModel
    func_tokenizer = T5Tokenizer
  elif "xlm-roberta-base" == model_name:
    func_model = XLMRobertaModel
    func_tokenizer = XLMRobertaTokenizer
  elif "ernie-m-base_pytorch" == model_name:
    func_model = ErnieMModel
    func_tokenizer = ErnieMTokenizer

  print(f"Using {func_model.__name__} with {func_tokenizer.__name__}")
  
  model_path = "Models/"
  tokenizer = func_tokenizer.from_pretrained(model_path + model_name, use_fast=False)
  model = func_model.from_pretrained(model_path + model_name)
  model = model.to(device) # move to GPU

  return tokenizer, model

def pool_embeddings(method, data, tokenized, pad_tok_id):
  if "attention_mask" in tokenized:
    attention_mask = tokenized["attention_mask"]
  else: # apparently ErnieM does NOT have attenion IDs in the tokenized output, so I am "computing" them myself - like in all other models, the model should not pay attention to [PAD] tokens, so they are ignored/not paid attention to
    token_ids = tokenized["input_ids"][0]
    padding_ids = len([tok for tok in token_ids if tok == pad_tok_id]) # count how many [PAD] tokens there are
    attention_mask = torch.ones((tokenized["input_ids"].shape)).to(device)
    if padding_ids > 0:
        attention_mask[:,-padding_ids:] = 0
    attention_mask = torch.tensor(attention_mask).to(device)
    
  attention_expanded = attention_mask.unsqueeze(-1).expand(data.size()).float()
  data_attention = data * attention_expanded
  return torch.sum(data_attention, 1) / torch.clamp(attention_expanded.sum(1), min=1e-9) # to not divide by 0

def get_embeddings(docs, lang, model_type, path):
  tokenizer, model = load_tokenizer_model(model_type)

  emb = NpyAppendArray(path + f"emb_{model_type}_{lang}.npy")

  for idx, d in enumerate(tqdm(docs, desc=f"{model_type} - {lang}")):
    inputs = tokenizer(d,return_tensors="pt",max_length=512,truncation=True,padding="max_length")
    
    pad_tok_id = tokenizer("[PAD]")
    pad_tok_id = pad_tok_id["input_ids"][1]
    
    aux = {}
    for key in inputs.keys(): # cast tokenized input to GPU
      aux[key] = inputs[key].to(device)
    inputs = aux

    outputs = model(**inputs)

    outputs = pool_embeddings(torch.mean, outputs[0], inputs, pad_tok_id)

    last_hidden_state = np.array(outputs.cpu().detach().numpy())  # The last hidden-state is the first element of the output tuple

    emb.append(last_hidden_state)
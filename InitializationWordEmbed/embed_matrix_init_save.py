# peft model py 681
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
# import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaTokenizer
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model 
from torch.nn import DataParallel
import math
import wandb
import numpy as np
import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--indic_tokenizer", default="indic_llama_hf_m_filter")
parser.add_argument("--source_tokenizer", default="./llama_fast_tokenizer/")
parser.add_argument("--strategy", default="intersect_random", help="intersect_random || intersect_wechsel || all_wechsel")
parser.add_argument("--block_size", type=int, default=512)
parser.add_argument('--model_config', default="./config_llama2/", type=str)
parser.add_argument('--model_path', default="./model_llama2/", type=str)
parser.add_argument('--save_model_config', default="./config_indicLLma_IR", type=str)
parser.add_argument('--save_model_path', default="./model_indicllama_IR/", type=str)
parser.add_argument('--wechsel_emb_path', default="/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/trained_fasttext/embed/indicllama_128k_wechsel_dict.pt", type=str)


args = parser.parse_args()


##############creating word embediing
source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer)
target_tokenizer = AutoTokenizer.from_pretrained(args.indic_tokenizer)

config = AutoConfig.from_pretrained(args.model_config, trust_remote_code=True)


source_model = AutoModelForCausalLM.from_pretrained(
  args.model_path,
  config=config,
  trust_remote_code=True,
)

embedding_size = source_model.get_input_embeddings().weight.shape[0]


source_matrix = source_model.get_input_embeddings().weight.detach().numpy().copy()
source_vocab = source_tokenizer.vocab
target_vocab = target_tokenizer.vocab

target_matrix = np.zeros(
        (len(target_tokenizer), source_matrix.shape[1]), dtype=source_matrix.dtype
    )
mean, std = (
        source_matrix.mean(0),
        source_matrix.std(0),
    )
random_fallback_matrix = np.random.RandomState(1234).normal(
        mean, std, (len(target_tokenizer.vocab), source_matrix.shape[1])
    )

if args.strategy == "intersect_random":
    print("in intersect_random")
    for index, token in enumerate(target_vocab):
        if token in source_vocab:
            target_matrix[target_vocab[token]] = source_matrix[source_vocab[token]]
        else :
            target_matrix[target_vocab[token]] = random_fallback_matrix[index]
elif args.strategy == "intersect_wechsel":
    print("in intersect wechsel")
    wechsel_matrix = torch.load(args.wechsel_emb_path)
    for index, token in enumerate(target_vocab):
        if token in source_vocab:
            target_matrix[target_vocab[token]] = source_matrix[source_vocab[token]]
        else :
            target_matrix[target_vocab[token]] = wechsel_matrix[index]

elif args.strategy == "all_wechsel":
    wechsel_matrix = torch.load(args.wechsel_emb_path)
    for index, token in enumerate(target_vocab):
        target_matrix[target_vocab[token]] = wechsel_matrix[index]


config.vocab_size = len(target_matrix)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
print(model)

model.state_dict()['model.embed_tokens.weight'].copy_(torch.from_numpy(target_matrix))   #word embedding setting
for param in source_model.state_dict():
    # print(param)
    if "model.embed_tokens.weight" in param:
        print("do nothing ", param )
    elif "lm_head.weight" in param:
        print("lm_head.weight")
    else :
        # print(param)
        model.state_dict()[param].copy_(source_model.state_dict()[param])

config.save_pretrained(args.save_model_config)
model.save_pretrained(args.save_model_path)

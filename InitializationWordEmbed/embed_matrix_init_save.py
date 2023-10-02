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
parser.add_argument("--strategy", default="intersect_wechsel", help="intersect_random_normal | intersect_random_multivariate_normal || intersect_wechsel || direct_init")
parser.add_argument("--multivariate_sigma_scale", type=float, default=1e-5)
parser.add_argument("--block_size", type=int, default=512)
parser.add_argument('--model_config', default="./config_llama2/", type=str)
parser.add_argument('--model_path', default="./model_llama2/", type=str)
parser.add_argument('--save_model_config', default="./config_indicLLma_IR_wikitionary_correct", type=str)
parser.add_argument('--save_model_path', default="./model_indicllama_IR_wikitionary_correct/", type=str)
parser.add_argument('--wechsel_emb_path', default="/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/trained_fasttext/embed/indicllama_128k_wechsel_dict_eikitionary.pt", type=str)
parser.add_argument('--wechsel_lmhead_path', default="/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/trained_fasttext/embed/indicllama_lmhead_128k_wechsel_dict_wikitionary.pt", type=str)



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


source_matrix_emb = source_model.get_input_embeddings().weight.detach().numpy().copy()
source_matrix_lmhead = source_model.state_dict()['lm_head.weight'].numpy()
source_vocab = source_tokenizer.vocab
target_vocab = target_tokenizer.vocab

target_matrix_emb = np.zeros(
        (len(target_tokenizer), source_matrix_emb.shape[1]), dtype=source_matrix_emb.dtype
    )

target_matrix_lmhead = np.zeros(
        (len(target_tokenizer), source_matrix_lmhead.shape[1]), dtype=source_matrix_lmhead.dtype
    )


if args.strategy == "intersect_random_normal":
    mean_emb, std_emb = (
            source_matrix_emb.mean(0),
            source_matrix_emb.std(0),
        )

    mean_lmhead, std_lmhead = (
            source_matrix_lmhead.mean(0),
            source_matrix_lmhead.std(0),
        )


    random_fallback_matrix_emb = np.random.RandomState(1234).normal(
            mean_emb, std_emb, (len(target_tokenizer.vocab), source_matrix_emb.shape[1])
        )

    random_fallback_matrix_lmhead = np.random.RandomState(1234).normal(
            mean_lmhead, std_lmhead, (len(target_tokenizer.vocab), source_matrix_lmhead.shape[1])
        )
elif args.strategy == "intersect_random_multivariate_normal":
    mu_emb = source_matrix_emb.mean(0)
    n_emb = source_matrix_emb.shape[0]
    sigma_emb = (source_matrix_emb - mu_emb).T.dot(source_matrix_emb - mu_emb) / n_emb
    mu_lmhead = source_matrix_lmhead.mean(0)
    n_lmhead = source_matrix_lmhead.shape[0]
    sigma_lmhead = (source_matrix_lmhead - mu_lmhead).T.dot(source_matrix_lmhead - mu_lmhead) / n_lmhead
    random_fallback_matrix_emb = np.random.multivariate_normal(mu_emb, args.multivariate_sigma_scale * sigma_emb, (len(target_tokenizer.vocab)))
    random_fallback_matrix_lmhead = np.random.multivariate_normal(mu_lmhead, args.multivariate_sigma_scale * sigma_lmhead, (len(target_tokenizer.vocab)))


if args.strategy == "intersect_random_normal" or args.strategy == "intersect_random_multivariate_normal":
    print("in", args.strategy)
    for index, token in enumerate(target_vocab):
        if token in source_vocab:
            target_matrix_emb[target_vocab[token]] = source_matrix_emb[source_vocab[token]]
        else :
            target_matrix_emb[target_vocab[token]] = random_fallback_matrix_emb[index]
        
        if token in source_vocab:
            target_matrix_lmhead[target_vocab[token]] = source_matrix_lmhead[source_vocab[token]]
        else :
            target_matrix_lmhead[target_vocab[token]] = random_fallback_matrix_lmhead[index]
elif args.strategy == "intersect_wechsel":
    print("in intersect wechsel")
    wechsel_matrix_emb = torch.load(args.wechsel_emb_path)
    wechsel_matrix_lmhead = torch.load(args.wechsel_lmhead_path)
    for index, token in enumerate(target_vocab):
        if token in source_vocab:
            target_matrix_emb[target_vocab[token]] = source_matrix_emb[source_vocab[token]]
        else :
            target_matrix_emb[target_vocab[token]] = wechsel_matrix_emb[index]
        
        if token in source_vocab:
            target_matrix_lmhead[target_vocab[token]] = source_matrix_lmhead[source_vocab[token]]
        else :
            target_matrix_lmhead[target_vocab[token]] = wechsel_matrix_lmhead[index]

elif args.strategy == "direct_init":
    wechsel_matrix_emb = torch.load(args.wechsel_emb_path)
    wechsel_matrix_lmhead = torch.load(args.wechsel_lmhead_path)
    target_matrix_emb = wechsel_matrix_emb
    target_matrix_lmhead = wechsel_matrix_lmhead


config.vocab_size = len(target_matrix_emb)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
print(model)

model.state_dict()['model.embed_tokens.weight'].copy_(torch.from_numpy(target_matrix_emb))   #word embedding setting
model.state_dict()['lm_head.weight'].copy_(torch.from_numpy(target_matrix_lmhead))
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

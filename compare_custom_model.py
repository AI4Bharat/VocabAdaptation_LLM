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
# import wandb
import numpy as np
import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--indic_tokenizer", default="/data-3/nandini/vocab_adapt/codes/indic_llama_hat_m_filter/")
parser.add_argument("--source_tokenizer", default="/data-3/nandini/vocab_adapt/codes/llama_fast_tokenizer/")
parser.add_argument("--strategy", default="intersect_wechsel", help="intersect_random_normal || intersect_random_multivariate_normal || intersect_wechsel || intersect_focus || direct_init ")
parser.add_argument("--multivariate_sigma_scale", type=float, default=1e-5)
parser.add_argument("--block_size", type=int, default=512)
parser.add_argument('--model_config', default="/data-3/nandini/vocab_adapt/codes/config_llama2/", type=str)
parser.add_argument('--model_path', default="/data-3/nandini/vocab_adapt/codes/model_llama2/", type=str)
parser.add_argument('--save_model_config', default="/data-3/nandini/vocab_adapt/codes/config_indicllama_IR_wechsel/", type=str)
parser.add_argument('--save_model_path', default="/data-3/nandini/vocab_adapt/codes/model_indicllama_IR_wechsel/", type=str)
parser.add_argument('--wechsel_emb_path', default="/data-3/nandini/vocab_adapt/codes/embed_wechsel/indicllama_embed_63k.pt", type=str)
parser.add_argument('--wechsel_lmhead_path', default="/data-3/nandini/vocab_adapt/codes/embed_wechsel/indicllama_lmhead_63k.pt", type=str)
parser.add_argument('--save_matrix_embed', default="/data-3/nandini/vocab_adapt/codes/embed_wechsel/indicllama_embed_63k.pt", type=str)
parser.add_argument('--save_matrix_lmhead', default="/data-3/nandini/vocab_adapt/codes/embed_wechsel/indicllama_lmhead_63k.pt", type=str)


args = parser.parse_args()


##############creating word embedding
source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer, use_fast= False)
target_tokenizer = AutoTokenizer.from_pretrained(args.indic_tokenizer, use_fast = False)

config = AutoConfig.from_pretrained(args.model_config, trust_remote_code=True)


source_model = AutoModelForCausalLM.from_pretrained(
  args.model_path,
  config=config,
  trust_remote_code=True,
)

target_model = AutoModelForCausalLM.from_pretrained(
  "/data-3/nandini/vocab_adapt/codes/minillama-frozen-embeds/model_size-7b1-total-batch-size-1024-batch-size-per-gpu-1-lr-5e-5-epochs-1-direct-llama-fft/"
)
print("target model is: ")
print(target_model)

embedding_size = source_model.get_input_embeddings().weight.shape[0]
print("embedding size is: ", embedding_size)


source_matrix_emb = source_model.get_input_embeddings().weight.detach().numpy().copy()
source_matrix_lmhead = source_model.state_dict()['lm_head.weight'].numpy()

target_matrix_emb = target_model.get_input_embeddings().weight.detach().numpy().copy()
target_matrix_lmhead = target_model.state_dict()['lm_head.weight'].numpy()

source_vocab = source_tokenizer.get_vocab()
target_vocab = target_tokenizer.get_vocab()





total_equal_embed = 0
total_not_equal_embed = 0
total_equal_lmhead = 0
total_not_equal_lmhead= 0
for index, token in enumerate(target_vocab):
    if token in source_vocab:
        if np.array_equal(target_matrix_emb[target_vocab[token]], source_matrix_emb[source_vocab[token]]):
            total_equal_embed +=1
        
        else :
            total_not_equal_embed += 1
        
        if np.array_equal(target_matrix_lmhead[target_vocab[token]], source_matrix_lmhead[source_vocab[token]]):
            total_equal_lmhead +=1
        
        else :
            total_not_equal_lmhead += 1

print("equal are embed: ", total_equal_embed)
print("not equal are embed: ", total_not_equal_embed)
print("equal are lmhead: ", total_equal_lmhead)
print("not equal are lmhead: ", total_not_equal_lmhead)

#  peft model py 681
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
import argparse

parser = argparse.ArgumentParser()
import transformers


# wandb.init(project="vocab_adap_clm", entity="nandinimundra", name = f"{args.run_name}")   
# indic_tokenizer = AutoTokenizer.from_pretrained(f"indic_tokenizer/mpt_tokenizer_before_128k")
indic_tokenizer = LlamaTokenizer.from_pretrained(f"indic_llama_hf_m")
# indic_tokenizer.save_pretrained(f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/indicBert_tokenizer")
base_tokenizer =  AutoTokenizer.from_pretrained("./llama_fast_tokenizer/")
print("length of indic tokenizer: ", len(indic_tokenizer))
print("length of llama tokenizer: ", len(base_tokenizer))


# source_vocab = base_tokenizer.vocab
# target_vocab = indic_tokenizer.vocab
# count = 0
# for index, token in enumerate(target_vocab):
#     if token in source_vocab:
#         count+=1
# print("intersection vocab ", count)

  


folders = ['eng_Latn-hin_Deva','eng_Latn-eng_Latn', 'eng_Latn-brx_Deva',  'eng_Latn-tam_Taml', 'eng_Latn-asm_Beng', 'eng_Latn-sat_Olck' ]
# for folder in os.listdir("/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/seed_train_test/"):
for folder in folders:
    list_base = []
    list_extended = []
    print(folder)
    text_path = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/seed_train_test/{folder}/test.{folder[-8:]}"
    
    with open(text_path,'r', encoding='utf-8' ) as file:
        data = file.readlines()
        word_set = set()
        count_mpt = 0
        count_indic = 0
        for line in data:
            words = line.split()
            word_set.update(words)
    
        for w in word_set:
            count_mpt += len(base_tokenizer.tokenize(w))
            count_indic += len(indic_tokenizer.tokenize(w))
        
        count_mpt = count_mpt/len(word_set)
        count_indic = count_indic/len(word_set)
        print("fertility score of llama is : ", count_mpt )
        print("fertility score of indic llama is : ", count_indic )
        

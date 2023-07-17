#  peft model py 681
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
# import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
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
indic_tokenizer = AutoTokenizer.from_pretrained(f"indic_tokenizer/indiMPT_tokenizer_64k")
base_tokenizer = tokenizer = AutoTokenizer.from_pretrained("./tokenizer_mpt_7b_model/")

source_vocab = base_tokenizer.vocab
target_vocab = indic_tokenizer.vocab
count = 0
for index, token in enumerate(target_vocab):
    if token in source_vocab:
        count+=1
print("intersection vocab ", count)
  


folders = ['eng_Latn-hin_Deva','eng_Latn-eng_Latn', 'eng_Latn-brx_Deva',  'eng_Latn-tam_Taml', 'eng_Latn-asm_Beng' ]
# for folder in os.listdir("/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/seed_train_test/"):
for folder in folders:
    list_base = []
    list_extended = []
    print(folder)
    text_path = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/seed_train_test/{folder}/test.{folder[-8:]}"
    with open(text_path,'r', encoding='utf-8' ) as file:
        data = file.readlines()
        len_data = len(data)
        for i in range(len_data):
            base = base_tokenizer(data[i])['input_ids']
            extended = indic_tokenizer(data[i])['input_ids']
            list_base.append(base)
            list_extended.append(extended)
    # print("inteersection of base and extended")
    print("total number of sentence ", len(list_base))
    count_mpt = 0
    count_indic = 0

    for j in range(len(list_base)):
        count_mpt += len(list_base[j])
        count_indic += len(list_extended[j])
        
    count_mpt = count_mpt / len(list_base)
    count_indic = count_indic / len(list_base)
    print("average mpt is : ", count_mpt )
    print("average indic mpt is : ", count_indic )
        
    # print("count of intersection is : ", count)

# peft model py 681
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
import wandb
import argparse

parser = argparse.ArgumentParser()
import transformers


parser.add_argument("--model_indic", default="indiMPT_tokenizer_64k")
parser.add_argument("--wordEmbTrain", default="false", help = "[true | false]")
parser.add_argument("--run_name", default="workshop")
args = parser.parse_args()
wandb.init(project="vocab_adap_clm", entity="nandinimundra", name = f"{args.run_name}")
name = 'mosaicml/mpt-7b'    
tokenizer = AutoTokenizer.from_pretrained(f"indic_tokenizer/{args.model_indic}")
config = AutoConfig.from_pretrained(f"./config_{args.model_indic}/", trust_remote_code=True)

# config.init_device = 'cuda:0' # For fast initialization directly on GPU!
model_kwargs = {"device_map": "auto"}
model = AutoModelForCausalLM.from_pretrained(
  f"./model_{args.model_indic}/",
  config=config,
#   load_in_8bit=True, 
#   torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True, **model_kwargs
)

def preprocess_function(examples):
    # result = tokenizer(examples["text"])
    # return result
    return tokenizer([" ".join(x) for x in examples["text"]])



def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

block_size = 512

# text_path =  'train_all_seed_corpus.txt'
# dataset = load_dataset("text", data_files=text_path)
# print(dataset)
# dataset = dataset.map(lambda samples: tokenizer(dataset['train']['text']), batched=True)
# tokenized_dataset = dataset['train'].map(
#         preprocess_function,
#         batched=True,
#         remove_columns=dataset["train"].column_names
#     )
# lm_dataset = tokenized_dataset.map(group_texts, batched=True)
# tokenizer.pad_token = tokenizer.eos_token
# model = DataParallel(model)

folders = ['eng_Latn-hin_Deva','eng_Latn-eng_Latn', 'eng_Latn-brx_Deva',  'eng_Latn-tam_Taml', 'eng_Latn-asm_Beng' ]
# for folder in os.listdir("/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/seed_train_test/"):
for folder in folders:
    text_path = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/seed_train_test/{folder}/test.{folder[-8:]}"
    dataset = load_dataset("text", data_files=text_path)
    print(folder , " " , dataset)
    tokenized_dataset = dataset['train'].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    lm_dataset = tokenized_dataset.map(group_texts, batched=True)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = transformers.TrainingArguments(
        output_dir= f"{args.run_name}_mpt_indic",
        evaluation_strategy="epoch",
        per_device_eval_batch_size = 16
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        eval_dataset=lm_dataset,
        data_collator=data_collator,
    )
    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results['eval_loss'])
    print(f"Perplexity- {folder[-8:]}: ", perplexity)
    wandb.log({"perp-{}-".format(folder[-8:]): perplexity})

# #################### Now evaluation

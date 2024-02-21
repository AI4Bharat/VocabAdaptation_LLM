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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
import torch
from tqdm import tqdm
import numpy as np
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from datasets import load_from_disk
import math
import wandb
import argparse
from datasets import load_from_disk

parser = argparse.ArgumentParser()
import transformers



parser.add_argument("--wordEmbTrain", default="true", help = "[true | false]")
parser.add_argument("--run_name", default="config_indicllama_IR_wechsel")
parser.add_argument('--model_config', default="/data/nandini/vocab_adapt/codes/config_indicllama_IR_wechsel/", type=str)
parser.add_argument('--model_path', default="/data/nandini/vocab_adapt/codes/model_indicllama_IR_wechsel/", type=str)
parser.add_argument('--tokenizer_path', default="/data/nandini/vocab_adapt/codes/indic_llama_hat_m_filter", type=str)
args = parser.parse_args()

wandb.init(project="vocabulary_adaptation_strategies", entity="nandinimundra", name = f"{args.run_name}")
   
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast = False)
config = AutoConfig.from_pretrained(args.model_config, trust_remote_code=True)

# config.init_device = 'cuda:0' # For fast initialization directly on GPU!
model_kwargs = {"device_map": "auto"}
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    config=config,
    #   load_in_8bit=True, 
    # torch_dtype=torch.bfloat16, # Load model weights in bfloat16
    trust_remote_code=True, **model_kwargs
)
print(model)


for name, param in model.named_parameters():
    param.requires_grad = False  # freeze the model - train adapters later
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)



model.enable_input_require_grads()

# for name, param in model.named_parameters():
#     print(f"{name} : ", param.size() )

# class CastOutputToFloat(nn.Sequential):
#   def forward(self, x): return super().forward(x).to(torch.float32)
# model.transformer.wte = CastOutputToFloat(model.transformer.wte)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj", "v_proj" ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)


if args.wordEmbTrain == "true":
    print("inside word emb train ")
    for name, param in model.named_parameters():
        if name == "base_model.model.lm_head.weight":
            print("inside param true ")
            param.requires_grad = True
        elif name == "base_model.model.model.embed_tokens.weight":
            print("inside param true ")
            param.requires_grad = True
    # print(f"Parameter name: {name}, Requires gradient: {param.requires_grad}")
    

for name, param in model.named_parameters():
    print(f"Parameter name: {name}, Requires gradient: {param.requires_grad}")

print_trainable_parameters(model)


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

block_size = 2048

text_path =  "/data/nandini/vocab_adapt/codes/lora_train_data_30M.txt"
dataset = load_dataset("text", data_files=text_path)
print(dataset)
# dataset = dataset.map(lambda samples: tokenizer(dataset['train']['text']), batched=True)
tokenized_dataset = dataset['train'].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
lm_dataset = tokenized_dataset.map(group_texts, batched=True)
tokenizer.pad_token = tokenizer.eos_token
# model = DataParallel(model)
trainer = transformers.Trainer(
    model=model, 
    train_dataset=  lm_dataset, # dataset['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_ratio=0.1, 
        # max_steps=200, 
        learning_rate=2e-4, 
        num_train_epochs=5,
        fp16=True,
        logging_strategy= 'epoch', 
        save_strategy="epoch",
        output_dir=args.run_name
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

model.config.use_cache = True



lang_data = ["english" , "hindi", "tamil", "assamese"]

for lang in lang_data:
    train_test_split = load_from_disk(f"/data-3/nandini/hf_dataset_split/{lang}")
    test = train_test_split['test']
    print("dataset is : ", lang)
    print(test)
   
    def calculate_perplexity(text):
        bos_token = tokenizer.bos_token #if tokenizer.bos_token else "<bos>"
        eos_token = tokenizer.eos_token #if tokenizer.eos_token else "<eos>"
        # print("bos_token is: ", bos_token, " eos_token is: ", eos_token)
        text_with_special_tokens = f"{bos_token} {text} {eos_token}"
        # print(text_with_special_tokens)
        encodings = tokenizer(text_with_special_tokens, return_tensors="pt")
        
        seq_len = encodings.input_ids.size(1)
        max_length = 4096
        stride = 512
        nlls = []
        prev_end_loc = 0

        nlls = []
        prev_end_loc = 0
        device = "cuda"
        for begin_loc in (range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        
        # print("nlls is: "  , nlls)

        # print("for ppl mean is: ",torch.stack(nlls).mean() )
        # print("for ppl median is: ", torch.stack(nlls).median())
        # print("for ppl mode is: ", torch.stack(nlls).mode())
        ppl = torch.exp(torch.stack(nlls).mean())
        
        # print("ppl is:  ", ppl)
        return ppl

    ppl_list = []
    count = 0

    # Calculating perplexity for each paragraph
    for i, paragraph in enumerate(tqdm(test)):
        # print(paragraph)
        ppl = calculate_perplexity(paragraph['text'])
        if(torch.isnan(ppl)):
            # print(paragraph['text'])
            count +=1
        else:
            ppl_list.append(ppl)

        # sum_perp += ppl

        # print(f"Paragraph {i+1}: Perplexity = {ppl}")
    len_ppl = len(ppl_list)
    # print("length of split is: ", cnt_split)
    print("count of nan value is : ",count)
    # avg_perp = sum_perp/len_ppl
    perplexity = torch.stack(ppl_list).mean()
    print("average is: ", perplexity)
    wandb.log({"perp-{}".format(lang): perplexity})

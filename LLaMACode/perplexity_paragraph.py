from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
import torch
import argparse
from tqdm import tqdm
import numpy as np
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

parser = argparse.ArgumentParser()
parser.add_argument("--max_length", type=int, default=2048)
parser.add_argument("--stride", type=int, default=512)
parser.add_argument('--precision', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'],
                    help='Precision format (fp32, fp16, bf16)')
parser.add_argument('--use_flash_attn', type=str, default='true', choices=['true', 'false'],
                    help='attention flag that is true or false')
args = parser.parse_args()

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('/data/nandini/vocab_adapt/codes/llama_tokenizer')
config = AutoConfig.from_pretrained("/data/nandini/vocab_adapt/codes/config_llama2/", trust_remote_code=True)
print("config loaded")

if args.precision == 'fp16':
    torch_dtype = torch.float16
elif args.precision == 'bf16':
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float32

device = "cuda"
model_kwargs = {"device_map": "auto"}
model = AutoModelForCausalLM.from_pretrained(
    "/data/nandini/vocab_adapt/codes/model_llama2/",
    config=config, 
    torch_dtype=torch_dtype,
    use_flash_attention_2=  True if "true" in args.use_flash_attn else False,
    **model_kwargs,
)
# model.to(device)
print("model loaded")

cache_directory = "/data-2/nandini/hf_dataset"
dataset = load_dataset("parquet", data_files='/data-2/nandini/english', cache_dir=cache_directory)
full_dataset = dataset['train']

# Split the dataset into train and test sets (10% for test set)
train_test_split = full_dataset.train_test_split(test_size=0.001)
test = train_test_split['test']
# test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# test = dataset.train_test_split(test_size=0.1)

def calculate_perplexity(text):
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    max_length = args.max_length 
    stride = args.stride  
    nlls = []
    prev_end_loc = 0

    nlls = []
    prev_end_loc = 0
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

    ppl = torch.exp(torch.stack(nlls).mean())
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
print("count of nan value is : ",count)
# avg_perp = sum_perp/len_ppl
print("average is: ", torch.stack(ppl_list).mean())

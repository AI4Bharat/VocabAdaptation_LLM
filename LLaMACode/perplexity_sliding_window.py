from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaTokenizer
from datasets import load_dataset
import torch
import argparse

parser = argparse.ArgumentParser()


# parser.add_argument("--wordEmbTrain", default="false", help = "[true | false]")
parser.add_argument("--max_length", type = int, default = 4096)
parser.add_argument("--stride", type=int, default=512)
parser.add_argument('--precision', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'],
                    help='Precision format (fp32, fp16, bf16)')
args = parser.parse_args()


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
  config=config, **model_kwargs,
  torch_dtype=torch_dtype, # Load model weights in bfloat16
#   trust_remote_code=True,
)
print("model loaded")


test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# print(test)
# print(test['text'][1])
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

import torch
from tqdm import tqdm

max_length = args.max_length 
print("max_length is: ", max_length)
stride = args.stride  
seq_len = encodings.input_ids.size(1)
print("sequence length is: ", seq_len)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
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
print(ppl)

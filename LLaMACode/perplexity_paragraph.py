from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
import torch
import argparse
from tqdm import tqdm
import numpy as np
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from datasets import load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument("--max_length", type=int, default=4096)
parser.add_argument("--stride", type=int, default=512)
parser.add_argument('--precision', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'],
                    help='Precision format (fp32, fp16, bf16)')
parser.add_argument('--use_flash_attn', type=str, default='true', choices=['true', 'false'],
                    help='attention flag that is true or false')
args = parser.parse_args()

# Initialize tokenizer and model

tokenizer = AutoTokenizer.from_pretrained('/data-3/nandini/vocab_adapt/codes/llama_tokenizer')
config = AutoConfig.from_pretrained("/data-3/nandini/vocab_adapt/codes/config_llama2/", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained('/data-3/nandini/vocab_adapt/codes/indic_llama_hat_m_filter/')
# config = AutoConfig.from_pretrained("/data-3/nandini/vocab_adapt/codes/config_indicllama_IR_wechsel/", trust_remote_code=True)
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
    "/data-3/nandini/vocab_adapt/codes/model_llama2/",
    # "/data-3/nandini/vocab_adapt/codes/model_indicllama_IR_wechsel/",
    config=config, 
    torch_dtype=torch_dtype,
    use_flash_attention_2=  True if "true" in args.use_flash_attn else False,
    **model_kwargs,
)
# model.to(device)
print("model loaded")

lang_data = ["english" , "hindi", "tamil", "assamese"]

for lang in lang_data:
    

    # cache_directory = "/data/nandini/hf_dataset"
    # dataset = load_dataset("parquet", data_files='/data/nandini/english', cache_dir=cache_directory)
    # # dataset = load_dataset("text", data_files='/data/nandini/indiccorp/as.txt', cache_dir=cache_directory)
    # full_dataset = dataset['train']

    # # Split the dataset into train and test sets (10% for test set)
    # train_test_split = full_dataset.train_test_split(test_size=0.0002)
    train_test_split = load_from_disk(f"/data/nandini/hf_dataset_split/{lang}")
    test = train_test_split['test']
    print("dataset is : ", lang)
    print(test)
    # test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # test = dataset.train_test_split(test_size=0.1)
    # cnt_split = 0

    def calculate_perplexity(text):
        bos_token = tokenizer.bos_token #if tokenizer.bos_token else "<bos>"
        eos_token = tokenizer.eos_token #if tokenizer.eos_token else "<eos>"
        # print("bos_token is: ", bos_token, " eos_token is: ", eos_token)
        text_with_special_tokens = f"{bos_token} {text} {eos_token}"
        # print(text_with_special_tokens)
        encodings = tokenizer(text_with_special_tokens, return_tensors="pt")
        
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
    print("average is: ", torch.stack(ppl_list).mean())

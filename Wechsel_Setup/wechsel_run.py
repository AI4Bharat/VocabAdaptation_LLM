import torch
from transformers import AutoModel, AutoTokenizer, BloomForCausalLM, AutoModelForCausalLM, AutoConfig, LlamaTokenizer
from datasets import load_dataset
import wechsel
from wechsel import WECHSEL, load_embeddings, WordEmbedding
from pathlib import Path
import fasttext
import fasttext.util
import tempfile
from functools import partial

print("hello 1")

source_tokenizer = AutoTokenizer.from_pretrained("llama_fast_tokenizer")
config = AutoConfig.from_pretrained("./config_llama2/")

model = AutoModelForCausalLM.from_pretrained(
  "./model_llama2/",
  config=config,
)

# for param in model.state_dict():
#     print(param)

output_hf_dir_m = 'indic_llama_hf_m_filter'
target_tokenizer = AutoTokenizer.from_pretrained(output_hf_dir_m)

# target_tokenizer = AutoTokenizer.from_pretrained("/home/nandini/vocab_adap/indicbloom_tokenizer_64k")
print("hello 2")
#model_target = fasttext.load_model("hin_Deva_fasttext.bin")


#embedding size - 4096



wechsel = WECHSEL(
    load_embeddings("/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/trained_fasttext/llama_tokenized_cbow.bin"),
    load_embeddings("/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/trained_fasttext/Indicllama_tokenize_128k_cbow.bin"),
    align_strategy= None,
)

print("hello 3")
## for LM head
# target_embeddings, info = wechsel.apply(
#     source_tokenizer,
#     target_tokenizer,
#     model.lm_head.weight.detach().numpy(),
    
# )
###############for main ###############
target_embeddings, info = wechsel.apply(
    source_tokenizer,
    target_tokenizer,
    model.get_input_embeddings().weight.detach().numpy(),
    
)

#print(model)
print(len(target_embeddings), "X", len(target_embeddings[0]))
torch.save(torch.from_numpy(target_embeddings), "/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/trained_fasttext/embed/indicllama_128k_wechsel.pt")
# model.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)
# model.config.vocab_size = len(target_embeddings)
# print(model)

# use `model` and `target_tokenizer` to continue training in Swahili!
# model = AutoModel.from_pretrained("bigscience/bloom-7b1")      #changed this

# text_path = f'/home/nandini/vocab_adap/data_tok/hin_Deva.txt'
# dataset = load_dataset("text", data_files=text_path, split="train")
# print(dataset)

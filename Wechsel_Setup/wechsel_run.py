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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--source_fasttext_bin', default="/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/trained_fasttext/llama_tokenized_cbow.bin", type=str)
parser.add_argument('--target_fasttext_bin', default="/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/trained_fasttext/Indicllama_tokenize_128k_cbow.bin", type=str)
parser.add_argument('--align_strategy', default=None, type=str, help="None || bilingual_dictionary") 
parser.add_argument('--bilingual_dictionary_path', default="/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/combine_bilingual_dict.txt", type=str)
parser.add_argument('--llama_tokenizer_dir', default="llama_fast_tokenizer", type=str)
parser.add_argument('--en_indic_tokenizer', default="indic_llama_hf_m_filter", type=str)
parser.add_argument('--model_config', default="./config_llama2/", type=str)
parser.add_argument('--model_path', default="./model_llama2/", type=str)
parser.add_argument('--output_path', default="/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/trained_fasttext/embed/indicllama_128k_wechsel_dict.pt", type=str)



args = parser.parse_args()

source_tokenizer = AutoTokenizer.from_pretrained(args.llama_tokenizer_dir)
config = AutoConfig.from_pretrained(args.model_config)

model = AutoModelForCausalLM.from_pretrained(
  args.model_path,
  config=config,
)

# output_hf_dir_m = 'indic_llama_hf_m_filter'
target_tokenizer = AutoTokenizer.from_pretrained(args.en_indic_tokenizer)

if args.align_strategy == None:
  wechsel = WECHSEL(
    load_embeddings(args.source_fasttext_bin),
    load_embeddings(args.target_fasttext_bin),
    align_strategy= None,
  )
else:
  wechsel = WECHSEL(
    load_embeddings(args.source_fasttext_bin),
    load_embeddings(args.target_fasttext_bin),
    align_strategy= "bilingual_dictionary",
    bilingual_dictionary = args.bilingual_dictionary_path,
)


target_embeddings, info = wechsel.apply(
    source_tokenizer,
    target_tokenizer,
    model.get_input_embeddings().weight.detach().numpy(),
    
)
print(len(target_embeddings), "X", len(target_embeddings[0]))
torch.save(torch.from_numpy(target_embeddings), args.output_path)

from transformers import AutoModel, AutoTokenizer, LlamaTokenizer
from datasets import load_dataset
from pathlib import Path
from functools import partial
import argparse

parser = argparse.ArgumentParser()

source_tokenizer = LlamaTokenizer.from_pretrained("llama_fast_tokenizer")

lang_txt = ["asm_Beng.txt", "ben_Beng.txt", "brx_Deva.txt", "doi_Deva.txt", "hin_Deva.txt",
            "gom_Deva.txt", "guj_Gujr.txt", "hin_Deva.txt", "kan_Knda.txt", "kas_Arab.txt", 
            "kas_Deva.txt", "mai_Deva.txt", "mal_Mlym.txt", "mar_Deva.txt", "mni_Beng.txt", 
            "mni_Mtei.txt", "npi_Deva.txt", "ory_Orya.txt", "pan_Guru.txt", "san_Deva.txt", 
            "sat_Olck.txt", "snd_Arab.txt", "snd_Deva.txt", "tam_Taml.txt", "tel_Telu.txt", "urd_Arab.txt"]

sent_list = []

for i in lang_txt:
    print(i[:-4])     
    with open(f'/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/data_tok/{i}', encoding='utf-8') as f:
        lines = f.readlines()
        sent_list.extend(lines)



out_file = open(f'/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/tokenized_data/llama_tokenized.txt', 'w', encoding='utf-8' )
cnt = 0
for line in sent_list:
    sent = line.strip()
    text = " ".join(source_tokenizer.tokenize(sent))
    out_file.write(text + "\n")
    cnt += 1
    if cnt%10000  == 0 :
        print(cnt)


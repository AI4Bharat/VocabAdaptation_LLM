from transformers import AutoTokenizer

from datasets import load_dataset
import pickle

#lang_txt = ["eng_Latn.txt", "hin_Deva.txt"]
#fasstext embedding for lang = ['Assamese', 'Bengali', 'Gujarati', 'Kannada']
lang_txt = ["asm_Beng.txt", "ben_Beng.txt", "brx_Deva.txt", "doi_Deva.txt", "eng_Latn.txt", 
            "gom_Deva.txt", "guj_Gujr.txt", "hin_Deva.txt", "kan_Knda.txt", "kas_Arab.txt", 
            "kas_Deva.txt", "mai_Deva.txt", "mal_Mlym.txt", "mar_Deva.txt", "mni_Beng.txt", 
            "mni_Mtei.txt", "npi_Deva.txt", "ory_Orya.txt", "pan_Guru.txt", "san_Deva.txt", 
            "sat_Olck.txt", "snd_Arab.txt", "snd_Deva.txt", "tam_Taml.txt", "tel_Telu.txt", "urd_Arab.txt"]

old_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")   #length of tokenizer is  50277
# print("length of tokenizer is ", len(old_tokenizer))
# source_tokenizer = AutoTokenizer.from_pretrained("/home/nandini/vocab_adap/indicbloom_tokenizer_v1")
sent_list = []
# for i in lang_txt:
#     print(i[:-4])     
#     with open(f'/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/data_tok/{i}', encoding='utf-8') as f:
        
#         lines = f.readlines()
#         for line in lines:
#             sent = line.strip()
#             sent_list.append(sent)

# file_metric = open(f'/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/data_tok/combine_data.txt', 'w', encoding='utf-8' )
# for k in range(0, len(sent_list)):
#     file_metric.write(sent_list[k])
#     file_metric.write("\n")

with open(f'/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/data_tok/combine_data.txt', encoding='utf-8' ) as f:
    lines = f.readlines()
    for line in lines:
        sent = line.strip()
        sent_list.append(sent)
    print(len(lines))

                
assert old_tokenizer.is_fast
tokenizer = old_tokenizer.train_new_from_iterator(sent_list, 100000)
tokenizer.save_pretrained(f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/indic_tokenizer/indiMPT_tokenizer_100k")


#text_path = f'/home/nandini/vocab_adap/data_tok/{i}'
    # dataset = load_dataset("text", data_files=text_path, split="train")
    # print(dataset['text'][:6])
    # print(dataset)   

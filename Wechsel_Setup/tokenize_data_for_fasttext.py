from transformers import AutoModel, AutoTokenizer, LlamaTokenizer
from datasets import load_dataset
# import wechsel
# from wechsel import WECHSEL, load_embeddings, WordEmbedding
from pathlib import Path
# import fasttext
# import fasttext.util
# import tempfile
from functools import partial
import argparse

parser = argparse.ArgumentParser()
# from wechsel.download_utils import download, gunzip
# parser.add_argument("--indic_tok", default="indiMPT_tokenizer_64k")
# args = parser.parse_args()
# source_tokenizer = AutoTokenizer.from_pretrained(f"indic_tokenizer/mpt_tokenizer_before_128k")
# output_hf_dir_m = 'indic_llama_hf_m_filter'
# source_tokenizer = AutoTokenizer.from_pretrained(f"./tokenizer_mpt_7b_model/")
source_tokenizer = LlamaTokenizer.from_pretrained("llama_fast_tokenizer")

lang_txt = ["asm_Beng.txt", "ben_Beng.txt", "brx_Deva.txt", "doi_Deva.txt", "hin_Deva.txt",
            "gom_Deva.txt", "guj_Gujr.txt", "hin_Deva.txt", "kan_Knda.txt", "kas_Arab.txt", 
            "kas_Deva.txt", "mai_Deva.txt", "mal_Mlym.txt", "mar_Deva.txt", "mni_Beng.txt", 
            "mni_Mtei.txt", "npi_Deva.txt", "ory_Orya.txt", "pan_Guru.txt", "san_Deva.txt", 
            "sat_Olck.txt", "snd_Arab.txt", "snd_Deva.txt", "tam_Taml.txt", "tel_Telu.txt", "urd_Arab.txt"]
# lang_txt = ["hin_Deva.txt"]


sent_list = []
# with open(f'train_all_indic_corp_10_optimal.txt', encoding='utf-8') as f:        
#     lines = f.readlines()
#     sent_list.extend(lines)

# print("after writting to file, length of sent_list after reading 10 lines: ", len(sent_list))
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
# out_file = '/data/nandini/tokenize_file/Indicbloom_tokenize.txt'
       
# model = fasttext.train_unsupervised(
#         out_file,
#         dim=4096,
#         neg=10,
#         model="cbow",
#         minn=5,
#         maxn=5,
#         epoch=10,
#         loss = 'hs',
#         wordNgrams = 5,
#         thread = 54,
#         lr = 0.01,
#     )
# # print(model)
# model.save_model("/data/nandini/fasttext_model/Indicbloom_fasttext.bin")

#model = AutoModel.from_pretrained("roberta-base")

# text_path = f'/home/nandini/vocab_adap/data_tok/eng_Latn.txt'
# # sent_list = []
# tok_list = []

# cnt = 0 
# with open(f'/home/nandini/vocab_adap/data_tok/eng_Latn.txt', encoding='utf-8') as f:
    
#     lines = f.readlines()
#     for line in lines:
#         sent = line.strip()
#         tok_list.append(" ".join(source_tokenizer.tokenize(sent)))
#         cnt += 1
#         if cnt>100:
#             break

# out_file = open(f'/home/nandini/vocab_adap/demo_1.txt', 'w', encoding='utf-8' )
# # out_file = tempfile.NamedTemporaryFile("w+", encoding='utf-8')
# count = 0
# print("token input for training fasttext model")
# for text in tok_list:
#     # print(text)
#     out_file.write(text + "\n")
#     count = count +1
#     if(count>100):
#         break
# print(out_file.name)
# # dataset = load_dataset("text", data_files=text_path, split="train")
# # print(dataset)
# # dataset = dataset.shuffle(seed=42)

# # dataset = dataset.map(
# #         lambda row: {"text": " ".join(source_tokenizer.tokenize(row["text"]))},
# #     )
# gjhgjh
# out_file = open(f'/home/nandini/vocab_adap/demo_1.txt', 'w', encoding='utf-8' )
# # out_file = tempfile.NamedTemporaryFile("w+", encoding='utf-8')
# count = 0
# print("token input for training fasttext model")
# for text in dataset["text"]:
#     # print(text)
#     out_file.write(text + "\n")
#     count = count +1
#     if(count>100):
#         break

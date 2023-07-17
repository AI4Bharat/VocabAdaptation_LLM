from transformers import AutoModel, AutoTokenizer, AutoConfig
import numpy as np
import torch
import json

source_tokenizer = AutoTokenizer.from_pretrained("./tokenizer_mpt_7b_model/")
target_tokenizer = AutoTokenizer.from_pretrained("indic_tokenizer/indiMPT_tokenizer_64k") 
#here target tokenizer is the vocab of what we are extending to the base toeknizer

source_vocab = source_tokenizer.vocab
target_vocab = target_tokenizer.vocab

########## contain list of all intersecting token
intersect= []
# f = open('indic_tokenizer/IndicMPTexplicit/tokenizer.json')
# data = json.load(f)
# for index, token in enumerate(target_vocab):
#     if token in source_vocab:
#         intersect.append(token) 

f = open('indic_tokenizer/indiMPT_tokenizer_64k/tokenizer.json')
data = json.load(f)
f_source = open('tokenizer_mpt_7b_model/tokenizer.json')
data_source = json.load(f_source)
vocab_dict = data['model']['vocab']
merge_source = data['model']['merges']
merge_target = data_source['model']['merges']
count = 50254
for index, token in enumerate(target_vocab):
    if token in source_vocab:
        intersect.append(token)
    else:
        data_source['model']['vocab'][token] = count
        count = count + 1
    # if count == 50256:
    #     break

print("length of source merges is ", len(merge_source))
print("length of target merges is ", len(merge_target))
print("merge type is ", merge_source[0])
merge_intersect = 0
for merge_pair in merge_target:
    if merge_pair in merge_source:
        merge_intersect += 1
    else:
         data['model']['merges'].append(merge_pair)
    # if merge_intersect == 1:
    #     break
print("intersected merge is ", merge_intersect)
print(len(merge_source))
print(len(data['model']['merges']))

print(data_source.keys())
with open("sample.json", "w",encoding='utf-8') as outfile:
    json.dump(data_source, outfile, indent = 3,ensure_ascii=False)



# print(vocab_dict)
# print(len(vocab_dict))

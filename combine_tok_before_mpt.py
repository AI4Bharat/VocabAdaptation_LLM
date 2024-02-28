from transformers import AutoModel, AutoTokenizer, AutoConfig
import numpy as np
import torch
import json
import argparse

parser = argparse.ArgumentParser()


# parser.add_argument("--wordEmbTrain", default="false", help = "[true | false]")
parser.add_argument("--indic_tok", default="indiMPT_tokenizer_64k")
args = parser.parse_args()
target_tokenizer = AutoTokenizer.from_pretrained("./tokenizer_mpt_7b_model/")
source_tokenizer = AutoTokenizer.from_pretrained(f"indic_tokenizer/{args.indic_tok}")

target_vocab = target_tokenizer.vocab
source_vocab = source_tokenizer.vocab


########## contain list of all intersecting token
intersect= []
# f = open('indic_tokenizer/IndicMPTexplicit/tokenizer.json')
# data = json.load(f)
# for index, token in enumerate(target_vocab):
#     if token in source_vocab:
#         intersect.append(token) 

f_source = open(f"indic_tokenizer/{args.indic_tok}/tokenizer.json")
data_source = json.load(f_source)
f_target = open('tokenizer_mpt_7b_model/tokenizer.json')
data_target = json.load(f_target)
# vocab_dict = data['model']['vocab']
merge_source = data_source['model']['merges']
merge_target = data_target['model']['merges']
print("length of merge_source ", len(merge_source))
print("length of merge_target ", len(merge_target))
# count = 50254
count = len(target_tokenizer)
print("length of target tokenizer ", len(target_tokenizer))
for index, token in enumerate(source_vocab):
    if token in target_vocab:
        intersect.append(token)
    else:
        data_target['model']['vocab'][token] = count
        count = count + 1
    # if count == 50256:
    #     break


print("merge type is ", merge_source[0])
merge_intersect = 0
count_rev = 0
for merge_pair in merge_target:
    if merge_pair in  merge_source:
        merge_intersect += 1
    else:
        data_source['model']['merges'].append(merge_pair)
        # #### insert at begining ###
        # data_target['model']['merges'].insert(count_rev, merge_pair)
        # count_rev+= 1

    # if merge_intersect == 1:
    #     break
data_target['model']['merges'] = data_source['model']['merges']
print("intersected merge is ", merge_intersect)
# print(len(merge_source))
# print(len(data['model']['merges']))
print("should be same length of target merges is after: ", len(data_target['model']['merges']), len(merge_source))

# print(data_source.keys())
with open("tokenizer.json", "w",encoding='utf-8') as outfile:
    json.dump(data_target, outfile, indent = 3,ensure_ascii=False)



# print(vocab_dict)
# print(len(vocab_dict))

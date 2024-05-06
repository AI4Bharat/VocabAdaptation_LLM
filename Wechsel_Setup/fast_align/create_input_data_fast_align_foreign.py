import os
from datasets import load_dataset


# for folder in os.listdir("/data-3/nandini/vocab_adapt/codes/bpcc_combine/"):
# dataset = load_dataset("xezpeleta/ccmatrix", "de-en",  cache_dir = '/data-3/nandini/nllb_dataset/', trust_remote_code=True)
dataset = load_dataset("xezpeleta/ccmatrix", "en-ru",  cache_dir = '/data-3/nandini/nllb_dataset/', trust_remote_code=True)
print("sharding the dataset")
total_examples = dataset['train'].num_rows
examples_per_shard = 10000000
num_shards = total_examples // examples_per_shard + (1 if total_examples % examples_per_shard > 0 else 0)

# Shard the dataset - selecting the first shard (index=0)
sharded_dataset = dataset['train'].shard(num_shards=num_shards, index=0)
print("sharded dataset is: ")
print(sharded_dataset)
data_en = []
data_fr = []
    
for translation in sharded_dataset['translation']:
    data_en.append(translation['en'])
    data_fr.append(translation['ru'])

if len(data_fr) == len(data_en):
    print("yes it is same")
else:
    print("no it is not")
print(len(data_en))


data_target=[]
for i in range(len(data_fr)):
    text_fr = data_fr[i]
    text_en = data_en[i]
    # print("fr language text is ", text_fr)
    # print("indian language text is ", text_en)
    t = text_en + " ||| " + text_fr + "\n"   #according to the input format of fastAlign
    data_target.append(t)
    

train_path = f"/data/nandini/vocab_adapt/codes/fast_align_data_new/merge.rus_Cyrl"
with open(train_path, 'w', encoding='utf-8') as train_file:
    train_file.writelines(data_target)

    

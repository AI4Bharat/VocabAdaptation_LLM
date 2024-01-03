from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
import json

# Load a BPE model from the provided vocab and merges
merge= []
final_line = []
count = 0
with open("tmp_dir_sentecepiece/merges.txt", 'r', encoding='utf-8') as file:
    for line in file.readlines():
        words = line.split()
        # print(words, " count is: ", count)
        # if words[1] == "''" or words[0] == "''":
        #     print("in word: ", words)
        
        if len(words) == 2:
            final_line.append(line)
        count+=1

        merge.extend(words)

with open("merge_2.txt", "w") as file:
    file.writelines(f"{item}" for item in final_line)

with open('tmp_dir_sentecepiece/vocab.json', 'r') as file:
    data = json.load(file)

for word in merge:
    if word not in data:
        print(word)
# njkhjkj
    

vocab, merges = models.BPE.read_file(vocab='tmp_dir_sentecepiece/vocab.json', merges='merge_2.txt')


# Initialize the tokenizer with the BPE model
bpe = models.BPE(vocab, merges)
tokenizer = Tokenizer(bpe)

# Add the required pre-tokenizer, decoder, and post-processor
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

# Save tokenizer to tokenizer.json
tokenizer.save('tmp_dir_sentecepiece/tokenizer.json')

print("Saved tokenizer to tokenizer.json")

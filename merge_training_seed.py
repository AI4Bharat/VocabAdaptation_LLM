import os
import random


merged_data = []
for folder in os.listdir("/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/seed_train_test/"):
    text_path = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/seed_train_test/{folder}/train.{folder[-8:]}"
    with open(text_path,'r', encoding='utf-8' ) as file:
        file_data = file.readlines()
        merged_data.extend(file_data)

# Shuffle the merged data randomly
random.shuffle(merged_data)
output_file = 'train_all_seed_corpus.txt'
with open(output_file, 'w', encoding='utf-8') as file:
    file.writelines(merged_data)
    # file.write('\n'.join(merged_data))

file_path = 'train_all_seed_corpus.txt'
with open(file_path,'r', encoding='utf-8' ) as file:
        data = file.readlines()

print(len(data))
# with open(file_path, 'r', encoding='utf-8') as file:
#     for line_number, line in enumerate(file, 1):
#         print(line)
#         if line_number == 20:
#             break

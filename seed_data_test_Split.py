from sklearn.model_selection import train_test_split
import os
from pathlib import Path


for folder in os.listdir("/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/BPCC-seed/"):
    # folder = 'eng_Latn-eng_Latn'
    # text_path = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/BPCC-seed/eng_Latn-hin_Deva/train.{folder[-8:]}"
    text_path = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/BPCC-seed/{folder}/train.{folder[-8:]}"
    with open(text_path,'r', encoding='utf-8' ) as file:
        data = file.readlines()

    # Split the data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    # Write the train data to a new file
    train_path = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/seed_train_test/{folder}/train.{folder[-8:]}"
    with open(train_path, 'w', encoding='utf-8') as train_file:
        train_file.writelines(train_data)

    # Write the test data to a new file
    test_path = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/seed_train_test/{folder}/test.{folder[-8:]}"

    with open(test_path, 'w', encoding='utf-8') as test_file:
        test_file.writelines(test_data)

    # Print the sizes of the train and test sets
    print("for file : ", folder[-8:])
    print(f"Train set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    

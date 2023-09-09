import os
from transformers import LlamaTokenizer, AutoTokenizer

## Save
output_hf_dir_m = 'indic_llama_hf_m_filter' # the path to save indic-LLaMA tokenizer

# print(f"indic-LLaMA tokenizer has been saved to {output_hf_dir_m}")

# Test
llama_tokenizer = AutoTokenizer.from_pretrained("./llama_fast_tokenizer/")
indic_llama_tokenizer = AutoTokenizer.from_pretrained(output_hf_dir_m)
for folder in os.listdir("/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/BPCC-seed/"):
    print(folder)
    # folder = 'eng_Latn-eng_Latn'
    # text_path = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/BPCC-seed/eng_Latn-hin_Deva/train.{folder[-8:]}"
    text_path_in = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/BPCC-seed/{folder}/train.{folder[-8:]}"
    text_path_en = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/BPCC-seed/{folder}/train.eng_Latn"
    with open(text_path_in,'r', encoding='utf-8' ) as file:
        data_in = file.readlines()
    
    with open(text_path_en,'r', encoding='utf-8' ) as file:
        data_en = file.readlines()
    
    data_target=[]
    for i in range(len(data_in)):
        sent_in = data_in[i].strip()
        text_in = " ".join(indic_llama_tokenizer.tokenize(sent_in))

        sent_en = data_en[i].strip()
        text_en = " ".join(llama_tokenizer.tokenize(sent_en))
        t = text_en + " ||| " + text_in + "\n"
        data_target.append(t)


    # Write the train data to a new file
    train_path = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/fastAlign_BPCC_tok/train_tok.{folder[-8:]}"
    with open(train_path, 'w', encoding='utf-8') as train_file:
        train_file.writelines(data_target)

    # # Write the test data to a new file
    
    # # Print the sizes of the train and test sets
    # print("for file : ", folder[-8:])
    # print(f"Train set size: {len(train_data)}")
    # print(f"Test set size: {len(test_data)}")

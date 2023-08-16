from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import os
# tokenizer_fast = AutoTokenizer.from_pretrained("./llama_fast_tokenizer/")
tokenizer = AutoTokenizer.from_pretrained("./llama_fast_tokenizer/")
file_metric = open(f'/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/check.txt', 'w', encoding='utf-8' )
for folder in os.listdir("/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/BPCC-seed/"):
    print(folder)
    cnt = 0
    # folder = 'eng_Latn-sat_Olck'
    # text_path = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/BPCC-seed/eng_Latn-sat_Olck/train.sat_Olck"
    text_path = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/BPCC-seed/{folder}/train.{folder[-8:]}"
    with open(text_path,'r', encoding='utf-8' ) as file:
        data = file.readlines()
        for lines in data:
            encoded_text = tokenizer(lines)['input_ids']
            decoded_text = tokenizer.decode(encoded_text)
            file_metric.write(lines)
            file_metric.write("\n")
            file_metric.write(decoded_text)
            file_metric.write("\n")
            # print(f"Tokenized :{tokenizer.tokenize(lines)}")   #to check the resulting token, generated tokens are mostly character level
            
            cnt += 1
            if cnt == 10:
                break
    break
    


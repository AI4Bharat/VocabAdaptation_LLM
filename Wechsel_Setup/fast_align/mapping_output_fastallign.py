import os
from transformers import LlamaTokenizer, AutoTokenizer

indic_llama_tokenizer = AutoTokenizer.from_pretrained(f"indic_llama_hf_m_filter")
llama_tokenizer =  AutoTokenizer.from_pretrained("./llama_fast_tokenizer/")

def map_tokens(source_sentence, target_sentence, alignment):
    source_tokens = source_sentence.split()
    target_tokens = target_sentence.split()
    mapping = []
    for align in alignment.split():
        src_idx, tgt_idx = map(int, align.split('-'))
        # print(src_idx, " : ",tgt_idx )
        if src_idx < len(source_tokens) and tgt_idx < len(target_tokens):
            src_token = source_tokens[src_idx]
            tgt_token = target_tokens[tgt_idx]
            mapping.append((src_token, tgt_token))

    return mapping


for folder in os.listdir("/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/BPCC-seed/"):
    # if folder == 'eng_Latn-eng_Latn' : 
    #     continue
    print(folder)
    # path to original parallel files
    text_path_in = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/BPCC-seed/{folder}/train.{folder[-8:]}"
    text_path_en = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/BPCC-seed/{folder}/train.eng_Latn"
    #path of files output by fastalign
    allign = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/fast_align/build/{folder[-8:]}.align"
    #path to save the resulted mapped files
    to_write = open(f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/align_dict/en_{folder[-8:]}.txt", 'w', encoding='utf-8')
    with open(text_path_in,'r', encoding='utf-8' ) as file:
        data_in = file.readlines()
    
    with open(text_path_en,'r', encoding='utf-8' ) as file:
        data_en = file.readlines()
    
    with open(allign,'r', encoding='utf-8' ) as file:
        alligned = file.readlines()
    
    data_target=[]
    for i in range(len(alligned)):
        sent_in = data_in[i].strip()
        text_in = " ".join(indic_llama_tokenizer.tokenize(sent_in))
        sent_en = data_en[i].strip()
        text_en = " ".join(llama_tokenizer.tokenize(sent_en))
        token_mapping = map_tokens(text_en, text_in, alligned[i])
        for src_token, tgt_token in token_mapping:
            to_write.write( src_token+ "\t" + tgt_token + "\n")
        

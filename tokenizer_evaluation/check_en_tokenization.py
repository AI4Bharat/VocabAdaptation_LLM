import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from transformers import LlamaTokenizer, AutoTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--llama_tokenizer_dir', default="llama_fast_tokenizer", type=str)
parser.add_argument('--indic_sp_model_file', default='/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/sp_indic.model', type=str)
args = parser.parse_args()

llama_tokenizer_dir = args.llama_tokenizer_dir
indic_sp_model_file = args.indic_sp_model_file

# load
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)



## Save
output_sp_dir_m = 'indic_llama_sp_m_filter'
output_hf_dir_m = 'indic_llama_hf_m_filter' # the path to save indic-LLaMA tokenizer

print(f"indic-LLaMA tokenizer has been saved to {output_hf_dir_m}")

# Test
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
indic_llama_tokenizer = AutoTokenizer.from_pretrained(output_hf_dir_m)

text_path = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/seed_train_test/eng_Latn-eng_Latn/test.eng_Latn"
cnt = 0
total_cnt = 0
with open(text_path,'r', encoding='utf-8' ) as file:
        data = file.readlines()
        # word_set = set()
        # count_mpt = 0
        # count_indic = 0
        # for line in data:
        #     words = line.split()
        #     word_set.update(words)
        
        # for w in word_set:
        #     total_cnt += 1
        #     llama_tok_res = llama_tokenizer.tokenize(w)
        #     indicllama_tok_res = indic_llama_tokenizer.tokenize(w)
            
        #     if indicllama_tok_res == llama_tok_res :
        #         cnt +=1
        #     else:
        #         print("Test text:\n",w)
        #         print(f"Tokenized by LLaMA tokenizer: ", llama_tok_res)
        #         print(f"Tokenized by indic-LLaMA tokenizer: ",indicllama_tok_res )
            # if total_cnt == 50:
            #      break
        

        
        #for sentence
        for text in data:
            total_cnt += 1
            llama_tok_res = llama_tokenizer.tokenize(text)
            indicllama_tok_res = indic_llama_tokenizer.tokenize(text)
            
            if indicllama_tok_res == llama_tok_res :
                cnt +=1
            else:
                print("Test text:\n",text)
                print(f"Tokenized by LLaMA tokenizer: ", llama_tok_res)
                print(f"Tokenized by indic-LLaMA tokenizer: ",indicllama_tok_res )
        #     if total_cnt == 50:
        #          break

ratio = total_cnt/cnt
print("total_cnt is: ", total_cnt)
print("cnt is: ", cnt)
print(ratio)
                  
                  

# import sentencepiece as spm

# vocab_file=output_sp_dir_m+'/indic_llama.model'
# sp_model = spm.SentencePieceProcessor()
# sp_model.Load("merged_tokenizer_sp_m/indic_llama.model")
# num_tokens = sp_model.GetPieceSize()

# Save the vocabulary to a .vocab file
# with open("output_vocab.vocab", "w", encoding="utf-8") as vocab_file:
#     for i in range(num_tokens):
#         token = sp_model.IdToPiece(i)
#         vocab_file.write(token + "\n")

# spm.SentencePieceTrainer.export_vocab(model = "merged_tokenizer_sp_m/indic_llama.model", output = "indic_llama_vocab")

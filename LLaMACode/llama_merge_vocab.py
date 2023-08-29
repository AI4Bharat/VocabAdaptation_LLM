import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse
import re
parser = argparse.ArgumentParser()
parser.add_argument('--llama_tokenizer_dir', default="llama_fast_tokenizer", type=str)
parser.add_argument('--indic_sp_model_file', default='/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/sp_indic.model', type=str)
args = parser.parse_args()

llama_tokenizer_dir = args.llama_tokenizer_dir
indic_sp_model_file = args.indic_sp_model_file

# load
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
indic_sp_model = spm.SentencePieceProcessor()
indic_sp_model.Load(indic_sp_model_file)

llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
# print(llama_spm)
# ghjgjh
# print("llama_spm is : ", llama_spm)
indic_spm = sp_pb2_model.ModelProto()
indic_spm.ParseFromString(indic_sp_model.serialized_model_proto())

# print number of tokens
print(len(llama_tokenizer),len(indic_sp_model))
print(llama_tokenizer.all_special_tokens)
print(llama_tokenizer.all_special_ids)
print(llama_tokenizer.special_tokens_map)


def not_contains_eng(text):
	eng_pattern = re.compile(r'[\u0020-\u007E]+')  # Range for english characters
	if eng_pattern.search(text):
		return False
	else:
		return True


## Add indic tokens to LLaMA tokenizer
llama_spm_tokens_set=set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before:{len(llama_spm_tokens_set)}")
for p in indic_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set and not_contains_eng(piece) :
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
    
print(f"New model pieces: {len(llama_spm.pieces)}")

## Save
output_sp_dir_m = 'indic_llama_sp_m_filter'
output_hf_dir_m = 'indic_llama_hf_m_filter' # the path to save indic-LLaMA tokenizer
os.makedirs(output_sp_dir_m,exist_ok=True)
with open(output_sp_dir_m+'/indic_llama.model', 'wb') as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir_m+'/indic_llama.model')

tokenizer.save_pretrained(output_hf_dir_m)
print(f"indic-LLaMA tokenizer has been saved to {output_hf_dir_m}")


# Test
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
indic_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir_m)

print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.special_tokens_map)
text="मैं वसई शहर का एक लड़का हूं इसलिए कृपया मुझे कुछ सिक्के दें ताकि मैं अपनी भूख मिटा सकूं"
print("Test text:\n",text)
print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
print(f"Tokenized by indic-LLaMA tokenizer:{indic_llama_tokenizer.tokenize(text)}")

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

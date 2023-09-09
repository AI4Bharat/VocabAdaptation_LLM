#! /bin/bash

./fasttext cbow -input /nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/tokenized_data/llama_tokenized.txt -output /nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/trained_fasttext/llama_tokenized_cbow -epoch 20 -thread 64 -dim 100 -neg 10  -minn 5 -maxn 5  -loss 'hs'
#  -wordNgrams 5  -thread 100

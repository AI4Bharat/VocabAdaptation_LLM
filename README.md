# Vocabulary Adaptation MPT and BLOOM model
<br>

## File specification
1. seed_data_test_split.py contains code to split seed dataset for train(90%) and test(10%)
2. merge_training_seed.py -> code to merge the training data
3. tokenizer_specification.py -> code to find how two tokenizer are related, such as intersecting token, or avg tokenization length per sentence
<br>
4. combine_tokenizer.py -> contains code to combine two tokenizer (The one used for extended version)
<br>
5. train_tokenizer.py -> train tokenizer from scratch
<br>
6. MPT_inference.py and IndicMPT_inference.py -> code to calculate the perplexity score of just inferncing(no training)
<br>
7. MPT_train.py and IndicMPT_train.py -> contains code to train LoRA adapetr and Word Embedding layer of model
<br>

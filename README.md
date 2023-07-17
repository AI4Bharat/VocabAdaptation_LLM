# Vocabulary Adaptation MPT and BLOOM model

## Result 
1. Please find result on https://docs.google.com/spreadsheets/d/1npkCffkNyztbPZokK9vis19zvzzT07l-uWnN06aiOeQ/edit#gid=868636088
2. Please find Meeting Notes/To-Do list/observation/.. on - https://docs.google.com/document/d/1dOegfXg8v5NBYXlCZgLDnkLBjP1YD_6K47kHh_5ojd0/edit

### File specification
1. seed_data_test_split.py contains code to split seed dataset for train(90%) and test(10%)
2. merge_training_seed.py -> code to merge the training data
3. tokenizer_specification.py -> code to find how two tokenizer are related, such as intersecting token, or avg tokenization length per sentence
4. combine_tokenizer.py -> contains code to combine two tokenizer (The one used for extended version)
5. train_tokenizer.py -> train tokenizer from scratch
6. MPT_inference.py and IndicMPT_inference.py -> code to calculate the perplexity score of just inferncing(no training)
7. MPT_train.py and IndicMPT_train.py -> contains code to train LoRA adapetr and Word Embedding layer of model


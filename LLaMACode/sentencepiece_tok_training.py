import sentencepiece as spm
# file_metric = open(f'/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/data_tok/combine_data_exclude_eng.txt', 'w', encoding='utf-8' )
file_train = '/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/train_all_indic_corp_10_optimal_BPCC.txt'
spm.SentencePieceTrainer.train(input=file_train, model_prefix='sp_indic', vocab_size=128000, model_type='bpe', max_sentence_length = 1073741824, shuffle_input_sentence='true', character_coverage = 1.0, num_threads = 64, hard_vocab_limit='false')
# spm.SentencePieceTrainer.train('--input=/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/train_all_indic_corp_10_optimal_BPCC.txt --model_prefix=sp_indic --vocab_size=128000 --model_type=bpe --max_sentence_length=1073741824')
sp_bpe = spm.SentencePieceProcessor()
sp_bpe.load('sp_indic.model')

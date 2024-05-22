model_name="/data/nandini/vocab_adapt/codes/merged_models/constraint_word2vec_1"
config_name="/data/nandini/vocab_adapt/codes/config_indicllama_constraint_word2vec"
tokenizer_name="/data/nandini/vocab_adapt/codes/indic_llama_hat_m_filter"
test_set="in22"
src_lang="hin_Deva"
tgt_lang="eng_Latn"
entity="nandinimundra"
project="MT_eval_few_shot"
cuda_available="0"

if [[ $src_lang == "eng_Latn" ]]; then
    dev_fname="data_flores/dev/flores_${src_lang}-${tgt_lang}.jsonl"
    test_fname="data_flores/test/${test_set}_${src_lang}-${tgt_lang}.jsonl"
else
    dev_fname="data_flores/dev/flores_${tgt_lang}-${src_lang}.jsonl"
    test_fname="data_flores/test/${test_set}_${tgt_lang}-${src_lang}.jsonl"
fi

declare -A batch_sizes
batch_sizes=( [1]=20 [4]=4 [8]=4 )
n_shots=(1 4 8)
# --------------------------------------------------------------------------
#                           Standard in-context eval
# --------------------------------------------------------------------------
for n_shot in "${n_shots[@]}"; do
    batch_size=${batch_sizes[$n_shot]}
    
    if [ -z "$batch_size" ]; then
        echo "Unsupported n_shot value: $n_shot"
        exit 1
    fi

    echo "Running standard in-context eval for ${src_lang}-${tgt_lang} for ${n_shot} shot"

    python3 causal_lm_eval.py \
        --model_name_or_path $model_name \
        --config_name_or_path $config_name \
        --tokenizer_name_or_path $tokenizer_name \
        --dev_fname $dev_fname \
        --test_fname $test_fname \
        --src_lang $src_lang --tgt_lang $tgt_lang \
        --entity $entity --project $project \
        --n_shot $n_shot --batch_size $batch_size --max_new_tokens 256 \
        --cuda_available_device $cuda_available \

    
done

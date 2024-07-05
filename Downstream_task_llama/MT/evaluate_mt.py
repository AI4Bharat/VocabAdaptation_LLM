import os
import json
import argparse
import random
import numpy as np
import pandas as pd
import wandb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaTokenizerFast, StoppingCriteria,AutoConfig 
from datasets import load_dataset, DatasetDict
from sacrebleu import sentence_bleu, corpus_bleu, sentence_chrf, corpus_chrf
from tqdm import tqdm
# from langcode2name import language_mapping
# from attack import apply_attack
# from utils import jaccard

language_mapping = {
    "hin_Deva": "Hindi",
    "eng_Latn": "English",
    "tam_Taml": "Tamil",
    "rus_Cyrl": "Russian",
    "deu_Latn": "German",
}

def initialize_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    # config = AutoConfig.from_pretrained(args.config_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        # config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(args.device)
    model.eval()

    return model, tokenizer


def compose_prompt(example, src_lang, tgt_lang, devset, k, seed, col_name="prompt"):
    src_col, tgt_col = f"sentence_{src_lang}", f"sentence_{tgt_lang}"

    prompt = f"Translate this from {language_mapping[src_lang]} into {language_mapping[tgt_lang]}:\n\n"

    if k > 0:
        # add few-shot in-context demonstrations
        demonstrations = devset.shuffle(seed=seed).select(range(k))
        for demonstration in demonstrations:
            prompt += f"{language_mapping[src_lang]}: {demonstration[src_col]}\n{language_mapping[tgt_lang]}: {demonstration[tgt_col]}\n\n"

    # add the test example
    prompt += f"{language_mapping[src_lang]}: {example[src_col]}\n{language_mapping[tgt_lang]}: "
    example[col_name] = prompt
    return example


def predict(args, batch_input_prompts, model, tokenizer):
    encodings = tokenizer(batch_input_prompts, padding=True, return_tensors="pt", truncation=True, max_length = 4096)
    encodings = encodings.to(args.device)

    with torch.inference_mode():
        batch_outputs = model.generate(
            **encodings,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stop_id_sequences=None,
        )

    batch_outputs = tokenizer.batch_decode(batch_outputs.detach().clone(), skip_special_tokens=True)
    batch_outputs = [
        output[len(prompt) :].strip().split("\n")[0] for prompt, output in zip(batch_input_prompts, batch_outputs)
    ]
    return batch_outputs


def main(args):
    # model_name = os.path.basename(args.model_name_or_path)
    path_dir = args.model_name_or_path
    child_dir = os.path.basename(path_dir)
    parent_path = os.path.dirname(path_dir)
    parent_dir = os.path.basename(parent_path)
    # run_name = f"{parent_dir}_{child_dir}"
    if parent_dir =="extended_llama_baselines_initialized":
        print("in initialized model")
        parent_dir = child_dir
        child_dir = "step_0"
    run_name = f"{parent_dir}_{child_dir}_{args.src_lang}_{args.tgt_lang}_{args.n_shot}"

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=run_name,
        save_code=True,
        config=vars(args),
    )

    print("loading data ...")
    dev_dataset = load_dataset("json", data_files=args.dev_fname)["train"]
    test_dataset = load_dataset("json", data_files=args.test_fname)["train"]

    dev_dataset = dev_dataset.remove_columns(
        [
            col
            for col in dev_dataset.column_names
            if col not in [f"sentence_{args.src_lang}", f"sentence_{args.tgt_lang}"]
        ]
    )
    test_dataset = test_dataset.remove_columns(
        [
            col
            for col in test_dataset.column_names
            if col not in [f"sentence_{args.src_lang}", f"sentence_{args.tgt_lang}"]
        ]
    )
    dataset = DatasetDict({"dev": dev_dataset, "test": test_dataset})

    dataset["test"] = dataset["test"].map(
        lambda x, i: compose_prompt(x, args.src_lang, args.tgt_lang, dataset["dev"], args.n_shot, i),
        with_indices=True,
    )

    input_prompts = dataset["test"]["prompt"]
    print(f"number of examples: {len(input_prompts)}")
    print(f"here are few examples: ")
    for i, input_prompt in enumerate(input_prompts[:5]):
        print(f"example {i + 1}")
        print(input_prompt)
        print()

    src_col, tgt_col = f"sentence_{args.src_lang}", f"sentence_{args.tgt_lang}"

    print("loading the model ...")
    model, tokenizer = initialize_model_and_tokenizer(args)
    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        )
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    
    model.config.pad_token_id = tokenizer.pad_token_id

    print("generating the outputs ...")
    hypotheses, references = [], []
    for start_idx in tqdm(range(0, len(input_prompts), args.batch_size)):
        end_idx = start_idx + args.batch_size
        batch_input_prompts = input_prompts[start_idx:end_idx]

        batch_hypotheses = predict(args, batch_input_prompts, model, tokenizer)
        batch_references = dataset["test"][tgt_col][start_idx:end_idx]

        hypotheses.extend(batch_hypotheses)
        references.extend(batch_references)

        if start_idx % 10 == 0:
            print(batch_hypotheses[0])

    print("computing bleu, chrf, chrf++ ...")
    bleu_score = corpus_bleu(hypotheses=hypotheses, references=[references]).score
    chrf_score = corpus_chrf(hypotheses=hypotheses, references=[references]).score
    chrf2_score = corpus_chrf(hypotheses=hypotheses, references=[references], word_order=2).score

    predictions = [
        {"prompt": prompt, "hypothesis": hypothesis, "reference": reference}
        for prompt, hypothesis, reference in zip(input_prompts, hypotheses, references)
    ]

    pred_table = wandb.Table(dataframe=pd.DataFrame(predictions))
    run.log(
        {
            "predictions": pred_table,
            "metrics/bleu": bleu_score,
            "metrics/chrf": chrf_score,
            "metrics/chrf2": chrf2_score,
        }
    )

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/data/nandini/vocab_adapt/codes/model_llama2",
        help="Name or path of the pre-trained language model.",
    )
    # parser.add_argument(
    #     "--config_name_or_path",
    #     type=str,
    #     default="/data/nandini/vocab_adapt/codes/config_llama2",
    #     help="Name or path of the pre-trained language model.",
    # )
    parser.add_argument(
        "--entity",
        type=str,
        default="nandinimundra",
        help="Wandb entity.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="MT_eval_few_shot",
        help="Wandb project.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="bigscience/bloom-7b1",
        help="Name or path of the tokenizer associated with the language model.",
    )
    parser.add_argument(
        "--test_fname",
        type=str,
        default="data_flores/test/flores_eng_Latn-hin_Deva.jsonl",
        help="Name or path of the test dataset for evaluation.",
    )
    parser.add_argument(
        "--dev_fname",
        type=str,
        default="data_flores/dev/flores_eng_Latn-hin_Deva.jsonl",
        help="Name or path of the development dataset for fine-tuning or validation.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Directory to cache the pre-trained language model and tokenizer.",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="eng_Latn",
        help="Source language code (e.g., eng_Latn for English in Latin script).",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="hin_Deva",
        help="Target language code (e.g., hin_Deva for Hindi in Devanagari script).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device type where the model should be loaded for inference.",
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=4,
        help="Number of in-context demonstrations for prompting.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for text generation and evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum numbers of tokens to generate, ignoring the number of tokens in the prompt during text generation.",
    )
    # parser.add_argument(
    #     "--cuda_available_device",
    #     type=str,
    #     default="0",
    #     help="Maximum numbers of tokens to generate, ignoring the number of tokens in the prompt during text generation.",
    # )
    
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir

    assert args.test_fname.endswith(".jsonl") and args.dev_fname.endswith(
        ".jsonl"
    ), "test and dev files should be jsonl."

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_available_device
    assert args.device == "cuda" and torch.cuda.is_available(), "No GPU device available for experiment."

    main(args)

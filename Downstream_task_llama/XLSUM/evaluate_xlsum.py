import argparse
import os
import random
import torch
import numpy as np
import pandas as pd
import time
import json
from tqdm import tqdm
import time
import evaluate
from datasets import load_dataset
from utils import (
    generate_completions,
    load_hf_lm_and_tokenizer,
    dynamic_import_function,
)
from bleurt import score
import wandb


def trim_context(context, max_context_length, tokenizer):
    tokenized_context = tokenizer.encode(context, add_special_tokens=False)
    if len(tokenized_context) > max_context_length:
        context = tokenizer.decode(
            tokenized_context[:max_context_length], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    return context


def format_example(text, lang, summary=None):
    prompt = f"{lang.capitalize()} article: {text}"
    prompt += f"\n{lang.capitalize()} summary:"
    if summary is not None:
        prompt += f" {summary}\n\n"
    return prompt


def gen_prompt(dev_data, lang, max_context_length, tokenizer, k=-1):
    prompt = f"Summarize the following {lang.capitalize()} article(s) as accurately as possible in few sentences.\n\n"
    if k > 0:
        exemplars = dev_data.select(range(k))
        for example in exemplars:
            prompt += format_example(
                text=trim_context(example["summary"], max_context_length=max_context_length, tokenizer=tokenizer),
                lang=lang,
                summary=example["text"],
            )
    return prompt


def main(args):
    path_dir = args.model_name_or_path
    child_dir = os.path.basename(path_dir)
    parent_path = os.path.dirname(path_dir)
    parent_dir = os.path.basename(parent_path)
    # run_name = f"{parent_dir}_{child_dir}"
    if parent_dir =="extended_llama_baselines_initialized":
        print("in initialized model")
        parent_dir = child_dir
        child_dir = "step_0"
    run_name = f"{parent_dir}_{child_dir}_{args.lang}_{args.ntrain}"

    run = wandb.init(project="llama_instruct_xlsum", entity="nandinimundra", name = f"{run_name}", save_code=True,config=vars(args),)
    random.seed(args.seed)    

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq,
        )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None

    if args.lang == "german":
        dataset = load_dataset("GEM/wiki_lingua", "de")
    else:
        dataset = load_dataset("csebuetnlp/xlsum", args.lang)
   
    if args.lang == "german":
        dataset = dataset.map(lambda x: {"summary": x["target"].strip()})
    else:
        dataset = dataset.map(lambda x: {"summary": x["title"].strip()})
    
    if args.lang == "german":
        dataset = dataset = dataset.map(lambda x: {"text": x["source"].strip()})
    else:
        dataset = dataset = dataset.map(lambda x: {"text": x["summary"].strip()})

    
    dev_data = dataset["validation"].select(range(min(len(dataset["validation"]), args.n_instances)))
    test_data = dataset["test"].select(range(min(len(dataset["test"]), args.n_instances)))

    prompts = []
    for i, example in enumerate(test_data):
        k = args.ntrain
        prompt_end = format_example(
            text=trim_context(example["text"], args.max_context_length, tokenizer), lang=args.lang
        )
        train_prompt = gen_prompt(dev_data, args.lang, args.max_context_length, tokenizer, k)
        prompt = train_prompt + prompt_end

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "The summary is: "
            else:
                prompt += " The summary is: "
        prompts.append(prompt)

    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=50,
        batch_size=args.eval_batch_size,
        stop_id_sequences=None,
    )
    # remove unnecessary space
    # print(outputs)
    outputs = [output.strip() for output in outputs]
    # exit()
    with open(os.path.join(args.save_dir, f"xlsum_{args.lang}_predictions.jsonl"), "w") as fout:
        for example, output in zip(test_data, outputs):
            example["prediction_text"] = output
            fout.write(json.dumps(example) + "\n")

    # flush all the GPU memory
    del model
    torch.cuda.empty_cache()
    import gc

    gc.collect()

    print("Calculating Rouge and BLEURT ...")
    rouge = evaluate.load("rouge")
    bleurt = score.BleurtScorer(args.bleurt_model_name_or_path)

    predictions = outputs
    # print(predictions)
    references = [example["summary"] for example in test_data]
    # print(references)

    rouge_metrics = rouge.compute(predictions=predictions, references=references)
    metrics = {
        "rouge1": rouge_metrics["rouge1"],
        "rouge2": rouge_metrics["rouge2"],
        "rougeL": rouge_metrics["rougeL"],
        "bleurt": np.mean(bleurt.score(candidates=predictions, references=references)),
    }
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    run.log(
        {   
            "metrics/rougeL": metrics["rougeL"],
            "metrics/rouge2": metrics["rouge2"],
            "metrics/rouge1": metrics["rouge1"],
            "metrics/bleurt": metrics["bleurt"],
        }
    )
    # save results
    # with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
    #     json.dump(metrics, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=4, help="number of examples to use for few-shot evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lang",
        type=str,
        default="german",
    )
    parser.add_argument("--save_dir", type=str, default="results/xlsum/llama-7B/")
    parser.add_argument(
        "--bleurt_model_name_or_path",
        type=str,
        default="/data/jaygala/bleurt/BLEURT-20",
        help="bleurt model to load for evaluation.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/data-3/nandini/extended_llama_baselines/llama_word2vec/step_124",
        help="if specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="/data-3/nandini/mt_eval/extended_llama_tokenizer",
        help="if specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--max_context_length", type=int, default=4096, help="maximum number of tokens in the context passage."
    )
    parser.add_argument(
        "--n_instances",
        type=int,
        default=1000,
        help="if specified, a maximum of n_instances will be used for the evaluation."
    )
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.",
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts.",
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`.",
    )
    args = parser.parse_args()
    main(args)

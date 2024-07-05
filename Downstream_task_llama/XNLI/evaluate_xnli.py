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
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils_xnli import (
    get_next_word_predictions,
    score_completions,
    load_hf_lm_and_tokenizer,
    dynamic_import_function,
)
import wandb

choices = ["true", "unknown", "false"]


def format_example(premise, hypothesis, label=None):
    prompt = "Premise: {premise}\nHypothesis: {hypothesis}".format(premise=premise, hypothesis=hypothesis)
    prompt += "\nAnswer:"
    if label is not None:
        prompt += " {label}\n\n".format(label=label)
    return prompt


def gen_prompt(dev_data, k=-1):
    prompt = f"Answer whether the hypothesis is more likely to be true (entailment), false (contradiction), or unknown (neutral) based on the given premise.\n\n"
    if k > 0:
        exemplars = dev_data.select(range(k))
        for example in exemplars:
            label = choices[example["label"]]
            prompt += format_example(premise=example["premise"], hypothesis=example["hypothesis"], label=label)
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

    run = wandb.init(project="llama_instruct_xnli", entity="nandinimundra", name = f"{run_name}", save_code=True,config=vars(args),)
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

    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)

    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    if args.lang == "ta":
        dataset = load_dataset("Divyanshu/indicxnli", 'ta')
    else:
        dataset = load_dataset("xnli", f"{args.lang}")
    
    dataset = dataset.map(lambda x: {"premise": x["premise"].strip()})
    dataset = dataset.map(lambda x: {"hypothesis": x["hypothesis"].strip()})
    dev_data = dataset["validation"]
    test_data = dataset["test"]

    prompts = []
    count = 0
    for i, example in enumerate(test_data):
        k = args.ntrain
        prompt_end = format_example(premise=example["premise"], hypothesis=example["hypothesis"])
        train_prompt = gen_prompt(dev_data, k)
        prompt = train_prompt + prompt_end

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "The answer is: "
            else:
                prompt += " The answer is: "

        tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        # make sure every prompt is less than 2048 tokens
        while len(tokenized_prompt) > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_data, k)
            prompt = train_prompt + prompt_end

            if args.use_chat_format:
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_formatting_function(messages, add_bos=False)
                if prompt[-1] in ["\n", " "]:
                    prompt += "The answer is: "
                else:
                    prompt += "The answer is: "

            tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids

        prompts.append(prompt)
    
    

    scoring_examples = []
    for prompt in prompts:
        scoring_example = {}
        scoring_example['prompt'] = prompt
        scoring_example['completions'] = choices
        scoring_examples.append(scoring_example)



    # get the answer for all examples
    # adding a prefix space here, as that's expected from the prompt
    # TODO: should raise a warning if this returns more than one token
    # answer_choice_ids = [tokenizer.encode(answer_choice, add_special_tokens=False)[-1] for answer_choice in choices]
    rolled_up_scores_diff, rolled_up_scores = score_completions(model,tokenizer,scoring_examples)
    predictions_with_prompt = {}

    for prompt, completion_scores in rolled_up_scores.items():
        best_completion = max(completion_scores, key=completion_scores.get)
        predictions_with_prompt[prompt] = best_completion

    best_completions = []


    for prompt,best_completion in predictions_with_prompt.items():
        best_completions.append(best_completion)




    ground_truths = [example["label"] for example in test_data]
    predictions = []


    for best_completion in best_completions:
        if best_completion == 'true':
            scored = 0
        elif best_completion == 'unknown':
            scored = 1
        elif best_completion == 'false':
            scored = 2
        
        predictions.append(scored)




    metrics = {
        "accuracy": accuracy_score(ground_truths, predictions),
        "precision": precision_score(ground_truths, predictions, average="macro"),
        "recall": recall_score(ground_truths, predictions, average="macro"),
        "f1": f1_score(ground_truths, predictions, average="macro"),
    }

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


    run.log(
        {
            # "predictions": pred_table,
            "metrics/accuracy": metrics["accuracy"],
            "metrics/precision": metrics["precision"],
            "metrics/recall": metrics["recall"],
            "metrics/f1": metrics["f1"],
        }
    )

    run.finish()

    # # save results
    # with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
    #     json.dump(metrics, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lang", type=str, default="hi", choices=["hi", "ta", "en", "ru", "de"]
    )
    parser.add_argument("--save_dir", type=str, default="results/indicxnli/llama-7B/")
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
    parser.add_argument("--eval_batch_size", type=int, default=8, help="batch size for evaluation.")
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

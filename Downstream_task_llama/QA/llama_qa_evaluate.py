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
import wandb

templates = {
    "with_context": (
        "Answer the following question based on the information in the given passage.",
        "Passage:",
        "Question:",
        "Answer:",
    ),
    "no_context": ("Answer the following question.", "Question:", "Answer:"),
}


def trim_context(context, max_context_length, tokenizer):
    tokenized_context = tokenizer.encode(context, add_special_tokens=False)
    if len(tokenized_context) > max_context_length:
        context = tokenizer.decode(
            tokenized_context[:max_context_length], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    return context


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

    run = wandb.init(project="llama_instruct_qa", entity="nandinimundra", name = f"{run_name}", save_code=True,config=vars(args),)
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

    if args.script == "native":
        if args.lang in ['en','de','ru','hi']:
            print('Hello')
            dataset = load_dataset("google/xquad", f"xquad.{args.lang}")
        else:
            dataset = load_dataset("ai4bharat/IndicQA", f"indicqa.{args.lang}")
    else:
        dataset = load_dataset("ai4bharat/IndicQA-romanized", f"indicqa.{args.lang}")

    dataset = dataset.map(lambda x: {"context": x["context"].strip()})
    dataset = dataset.map(lambda x: {"question": x["question"].strip()})
    if args.lang == 'ta':
        test_data = dataset["test"]
    else:
        test_data = dataset["validation"]

    prompts = []
    for i, example in enumerate(test_data):
        dev_data = test_data.filter(lambda x: x["question"] != example["question"]).shuffle(args.seed)
        k = args.ntrain

        if args.no_context:
            prompt, q_template, a_template = templates["no_context"]
            p_template = ""
        else:
            prompt, p_template, q_template, a_template = templates["with_context"]

        if k > 0:
            formatted_demo_examples = []
            exemplars = dev_data.select(range(k))
            for dev_example in exemplars:
                answer = (
                    "unanswerable" if dev_example["answers"]["text"][0] == "" else dev_example["answers"]["text"][0]
                )
                if args.no_context:
                    formatted_demo_examples.append(
                        q_template + " " + dev_example["question"] + "\n" + a_template + " " + answer
                    )
                else:
                    formatted_demo_examples.append(
                        p_template
                        + " "
                        + trim_context(dev_example["context"], args.max_context_length, tokenizer)
                        + "\n"
                        + q_template
                        + " "
                        + dev_example["question"]
                        + "\n"
                        + a_template
                        + " "
                        + answer
                    )
            prompt += "\n\n".join(formatted_demo_examples) + "\n\n"

        if args.no_context:
            prompt += q_template + " " + format(example["question"]) + "\n"
        else:
            prompt += (
                p_template
                + " "
                + format(trim_context(example["context"], args.max_context_length, tokenizer))
                + "\n"
                + q_template
                + " "
                + format(example["question"])
                + "\n"
            )

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            prompt += a_template if prompt[-1] in ["\n", " "] else " " + a_template
        else:
            prompt += a_template
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
    outputs = [output.strip().split("\n")[0] for output in outputs]

    with open(os.path.join(args.save_dir, f"indicqa_{args.lang}_predictions.jsonl"), "w") as fout:
        for example, output in zip(test_data, outputs):
            example["prediction_text"] = output
            fout.write(json.dumps(example) + "\n")

    print("Calculating F1, EM ...")
    metric = evaluate.load("squad")

    predictions = [{"id": example["id"], "prediction_text": output} for example, output in zip(test_data, outputs)]
    references = [{"id": example["id"], "answers": example["answers"]} for example in test_data]
    for i in range(len(references)):
        if references[i]["answers"]["text"][0] == "":
            references[i]["answers"]["text"][0] = "unanswerable"

    metrics = metric.compute(predictions=predictions, references=references)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    run.log(
        {
            # "predictions": pred_table,
            "metrics/exact_match": metrics["exact_match"],
            "metrics/f1": metrics["f1"],
        }
    )

    run.finish()

    # save results
    # with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
    #     json.dump(metrics, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=4, help="number of examples to use for few-shot evaluation.")
    parser.add_argument(
        "--no_context", action="store_true", help="If given, we're evaluating a model without the gold context passage."
    )
    parser.add_argument(
        "--max_context_length", type=int, default=2048, help="maximum number of tokens in the context passage."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lang", type=str, default = 'en'
    )
    parser.add_argument("--script", default="native", choices=["native", "roman"])
    parser.add_argument("--save_dir", type=str, default="results/indicqa/llama-7B/")
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

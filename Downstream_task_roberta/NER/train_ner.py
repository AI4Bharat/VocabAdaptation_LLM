from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch import nn
import copy
from datasets import load_dataset
from transformers import TrainingArguments
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer, EvalPrediction
from datasets import load_metric
import numpy as np
from transformers import EarlyStoppingCallback, IntervalStrategy
import argparse
import wandb
import os
#The labels for the NER task and the dictionaries to map the to ids or 
#the other way around
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--max_seq_length", type=int, default=256)
parser.add_argument('--model_config', default="./roberta_baseline/config_roberta_base/", type=str)
parser.add_argument('--model_path', default="./roberta_baseline/model_roberta_base/", type=str)
parser.add_argument('--tokenizer', default="FacebookAI/roberta-base", type=str)
args = parser.parse_args()
print(os.path.basename(args.model_path))

path_dir = args.model_path
child_dir = os.path.basename(path_dir)
parent_path = os.path.dirname(path_dir)
parent_dir = os.path.basename(parent_path)
run_name = f"{parent_dir}_{child_dir}"

# parts = path_dir.split(os.sep)
# desired_parts = parts[-3:]
# parent_parent_dir = desired_parts[0]
# parent_dir = desired_parts[1]
# child_dir = desired_parts[2]
# run_name = f"{parent_parent_dir}_{parent_dir}_{child_dir}"

# run_name = os.path.basename(args.model_path)

wandb.init(project="NER_Roberta", entity="nandinimundra", name = f"{run_name}")

labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
id_2_label = {id_: label for id_, label in enumerate(labels)}
label_2_id = {label: id_ for id_, label in enumerate(labels)}

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, add_prefix_space=True, use_fast = True)
config = AutoConfig.from_pretrained(args.model_config, num_labels=len(labels), label2id=label_2_id, id2label=id_2_label)
model = AutoModelForTokenClassification.from_pretrained(args.model_path, config=config)


pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
wandb.log({"no_of_parameter": pytorch_total_params})

#ar, bg, de, el, es, fr, hi, ru, sw, th, tr, ur, vi, zh
datasets = load_dataset('wikiann', 'en')
datasets = datasets.shuffle(seed=5)
dataset_de = load_dataset('wikiann', 'de')
dataset_hi = load_dataset('wikiann', 'hi')
dataset_ru = load_dataset('wikiann', 'ru')
dataset_ta = load_dataset('wikiann', 'ta')



training_args = TrainingArguments(
    f"/data-3/nandini/mlm_downstream_checkpoint/ner_checkpoint_train/{parent_dir}/{child_dir}",
    evaluation_strategy = "epoch", #IntervalStrategy.STEPS,
    save_strategy = "epoch",  #IntervalStrategy.STEPS,
    num_train_epochs=5,
    save_total_limit = 5,
    logging_steps=1000,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=args.weight_decay,
    warmup_ratio = 0.05,
    do_predict=True,
    #output_dir="ner_models/xlmr/",
    load_best_model_at_end=True,
    metric_for_best_model = 'eval_f1',
    greater_is_better = True,
    fp16=True,
    overwrite_output_dir=True,
    eval_steps = 1000,
)

# This method is adapted from the huggingface transformers run_ner.py example script 
# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples):
    text_column_name = "tokens"
    label_column_name = "ner_tags"
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding="max_length",
        truncation=True,
        max_length=512,
        is_split_into_words=True,   
    )
    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


test_dataset = datasets["test"]
dataset_tok = datasets.map(
    tokenize_and_align_labels,
    batched=True,
)

test_de = dataset_de['test'].map(tokenize_and_align_labels, batched=True,)
test_hi = dataset_hi['test'].map(tokenize_and_align_labels, batched=True,)
test_ru = dataset_ru['test'].map(tokenize_and_align_labels, batched=True,)
test_ta = dataset_ta['test'].map(tokenize_and_align_labels, batched=True,)

data_collator = DataCollatorForTokenClassification(tokenizer,)




# Metrics
metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    label_list = id_2_label

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_tok['train'],
    eval_dataset=dataset_tok['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    # callbacks = [EarlyStoppingCallback(early_stopping_patience= 3 )]
)

print(trainer.train())
print(trainer.evaluate())

print("########################### ZERO SHOT EVALUATION ###########################################")
total_accuracy = 0
language_count = 0

def eval_test_lang(data_test, data_name):
  global total_accuracy  # Tell Python to use the global variable
  global language_count  # Tell Python to use the global variable
  print("zero shot test performance for lang:   ", data_name )
  eval_trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    #args=TrainingArguments(output_dir="./eval_output", remove_unused_columns=False,),
    eval_dataset= data_test,
    )
  metric = eval_trainer.evaluate()
  print("the metric for language ", data_name , " is : ",  metric)
  wandb.log({"en-{}".format(data_name): metric})
  wandb.log({"en-{}-eval_loss".format(data_name): metric.get('eval_loss')})
  wandb.log({"en-{}-eval_precision".format(data_name): metric.get('eval_precision')})
  wandb.log({"en-{}-eval_recall".format(data_name): metric.get('eval_recall')})
  wandb.log({"en-{}-eval_f1".format(data_name): metric.get('eval_f1')})
  wandb.log({"en-{}-eval_accuracy".format(data_name): metric.get('eval_accuracy')})
  wandb.log({"en-{}-eval_runtime".format(data_name): metric.get('eval_runtime')})
  wandb.log({"en-{}-eval_samples_per_second".format(data_name): metric.get('eval_samples_per_second')})
  wandb.log({"en-{}-eval_steps_per_second".format(data_name): metric.get('eval_steps_per_second')})
  total_accuracy += metric.get('eval_f1')
  language_count += 1
  return

eval_test_lang(dataset_tok['test'], "en" )
eval_test_lang(test_de, 'de')
eval_test_lang(test_hi, 'hi')
eval_test_lang(test_ru, 'ru')
eval_test_lang(test_ta, 'ta')
average_accuracy = total_accuracy / language_count
wandb.log({"avg_f1": average_accuracy})

from datasets import  Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import  AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EvalPrediction
from transformers import EarlyStoppingCallback, IntervalStrategy
from transformers import AutoConfig
import numpy as np
from transformers import TrainerCallback
import argparse
import numpy as np
import wandb
import os


#os.environ["WANDB_MODE"] = "offline"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--model_name", default="roberta_base", help= "model type [xlmr-b | xlmr-l| indicbert]")
parser.add_argument('--model_config', default="./roberta_baseline/config_roberta_base", type=str)
parser.add_argument('--model_path', default="./roberta_baseline/model_roberta_base", type=str)
parser.add_argument('--tokenizer', default="FacebookAI/roberta-base", type=str)
parser.add_argument('--folder_init', default="nan", type=str)
args = parser.parse_args()

print(os.path.basename(args.model_path))

path_dir = args.model_path
child_dir = os.path.basename(path_dir)
parent_path = os.path.dirname(path_dir)
parent_dir = os.path.basename(parent_path)
    # parts = path_dir.split(os.sep)
    # desired_parts = parts[-3:]
    # parent_parent_dir = desired_parts[0]
    # parent_dir = desired_parts[1]
    # child_dir = desired_parts[2]
    # run_name = f"{parent_parent_dir}_{parent_dir}_{child_dir}"

# parent_dir = "initialized"
run_name = f"{parent_dir}_{child_dir}"
# run_name = os.path.basename(args.model_path)

# parts = args.model_path.split(os.sep)
# variant_checkpoint = parts[-2]
# variant_part = variant_checkpoint.split('_')[-2]  # splits and gets 'univariate'
# checkpoint_part = variant_checkpoint.split('_')[-1]  # splits and gets 'checkpoint-80000'

# # parts = args.model_path.split('_')
# # variant = parts[3]  # 'multivariate' is expected to be the fourth element
# # checkpoint = parts[-1] 
# run_name = f"{variant_part}_{checkpoint_part}"
id2label= {0: "entailment", 1: "neutral", 2: "contradiction"}
label2id = {"entailment":0, "neutral": 1, "contradiction":2 }


wandb.init(project="XNLI_Roberta", entity="nandinimundra", name = f"{run_name}")

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
config = AutoConfig.from_pretrained(
    args.model_config,
    num_labels=3,
    id2label={ 0: "entailment", 1: "neutral", 2: "contradiction"},
)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_path,
    config=config,
)

def preprocess(dataset):
  dataset = dataset.map(lambda batch: tokenizer.encode_plus( batch['premise'], batch['hypothesis'], max_length= 256, pad_to_max_length = True, truncation=True))
  dataset = dataset.rename_column("label", "labels")
  dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
  return dataset
  


dataset_en = load_dataset("xnli", 'en')
datset_en = dataset_en.shuffle(seed=5)
#dataset_en['train'] = dataset_en['train'].shard(num_shards=200, index=0)
dataset_de = load_dataset("xnli", 'de')
dataset_hi = load_dataset("xnli", 'hi')
dataset_ru = load_dataset("xnli", 'ru')
dataset_ta = load_dataset("Divyanshu/indicxnli", 'ta')


dataset_tok_en = preprocess(dataset_en)
# val_as = preprocess(dataset_as['validation'])


test_de = preprocess(dataset_de['test'])
test_hi = preprocess(dataset_hi['test'])
test_ru = preprocess(dataset_ru['test'])
test_ta = preprocess(dataset_ta['test'])


pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
wandb.log({"no_of_parameter": pytorch_total_params})



training_args = TrainingArguments(
    f"./data-3/nandini/mlm_downstream_checkpoint/xnli_checkpoint_train/{parent_dir}/{child_dir}",
    # f"./data-3/nandini/mlm_downstream_checkpoint/xnli_checkpoint_train/callbacks_xnli_roberta_{args.folder_init}_{run_name}",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=args.weight_decay,
    warmup_ratio = 0.1,
    evaluation_strategy = "epoch", #IntervalStrategy.STEPS,
    eval_steps = 1000,
    save_total_limit = 5,
    num_train_epochs= 5,
    save_strategy = "epoch", #IntervalStrategy.STEPS,
    logging_steps=1000,
    metric_for_best_model = 'eval_acc',
    greater_is_better = True,
    overwrite_output_dir=True,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    fp16=True,
)
wandb.log({"batch_size_train":training_args.per_device_train_batch_size })
def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  print("in compute accuracy ******************************************")
  return {"acc": (preds == p.label_ids).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= dataset_tok_en["train"],
    eval_dataset= dataset_tok_en["validation"],
    compute_metrics=compute_accuracy,
    # callbacks = [EarlyStoppingCallback(early_stopping_patience= 3 )]
  )

print(trainer.train())
print(trainer.evaluate())

total_accuracy = 0
language_count = 0

def eval_test_lang(data_test, data_name):
  global total_accuracy  # Tell Python to use the global variable
  global language_count  # Tell Python to use the global variable
  print("zero shot test performance for lang:   ", data_name )
  eval_trainer = Trainer(
    model=model,
    args= training_args, #TrainingArguments(output_dir="./eval_output", remove_unused_columns=False,),
    eval_dataset= data_test,
    compute_metrics=compute_accuracy,
    # callbacks=[CustomWandbCallback]
    )
  metric = eval_trainer.evaluate()
  print("the metric for language ", data_name , " is : ",  metric)
  wandb.log({"en-{}".format(data_name): metric})
  wandb.log({"en-{}-eval_loss".format(data_name): metric.get('eval_loss')})
  wandb.log({"en-{}-eval_acc".format(data_name): metric.get('eval_acc')})
  wandb.log({"en-{}-eval_runtime".format(data_name): metric.get('eval_runtime')})
  wandb.log({"en-{}-eval_steps_per_second".format(data_name): metric.get('eval_steps_per_second')})
  total_accuracy += metric.get('eval_acc', 0)
  language_count += 1
  return

eval_test_lang(dataset_tok_en['test'], "en_test" )
eval_test_lang(test_de, "de") 
eval_test_lang(test_hi, "hi") 
eval_test_lang(test_ru, "ru") 
eval_test_lang(test_ta, "ta") 
average_accuracy = total_accuracy / language_count
wandb.log({"avg_acc": average_accuracy})
eval_test_lang(dataset_tok_en['validation'], "en_val" )

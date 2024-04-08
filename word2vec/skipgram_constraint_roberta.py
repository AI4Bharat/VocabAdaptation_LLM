
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaTokenizer, AutoModelForMaskedLM
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import math
import torch
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import gc
import random
from collections import Counter
import wandb
import argparse
from datasets import load_from_disk
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
import torch
from tqdm import tqdm
from fuzzy_overlap_copy import modify_overlapping_tokens

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=1)
parser.add_argument("--dataset_file", type=str, default="code_mix_data_tokenization_2M")
parser.add_argument('--model_config', default="/data/nandini/vocab_adapt/codes/BERT/roberta_baseline/config_roberta_base/", type=str)
parser.add_argument('--model_path', default="/data/nandini/vocab_adapt/codes/BERT/roberta_baseline/model_roberta_base/", type=str)
parser.add_argument('--learning_rate', type=float, default=0.0003)
parser.add_argument('--batch_size',  type=int, default=4096)
parser.add_argument('--epoch',  type=int, default=5)
parser.add_argument('--mlm_probability',  type=int, default=0.15)
parser.add_argument('--window_size',  type=int, default=5)


args = parser.parse_args()


wandb.init(project="constrained_word2vec_train_roberta", entity="nandinimundra", name = f"{args.dataset_file}_{args.threshold}_{args.batch_size}_{args.learning_rate}_{args.window_size}_{args.mlm_probability}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

source_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
target_tokenizer = AutoTokenizer.from_pretrained('/data/nandini/vocab_adapt/codes/BERT/extended_llama_roberta/', use_fast = False)
print("in skipgram constraint roberta py file ")
target_to_source_dict = modify_overlapping_tokens(target_tokenizer = target_tokenizer  ,
                           source_tokenizer = source_tokenizer  , fuzzy_search=True, fuzzy_whitespace=True)


print("in the skipgram_constraint_roberta.py file ", len(target_to_source_dict))



config = AutoConfig.from_pretrained("/data/nandini/vocab_adapt/codes/BERT/roberta_baseline/config_roberta_base/")
source_model = AutoModelForMaskedLM.from_pretrained("/data/nandini/vocab_adapt/codes/BERT/roberta_baseline/model_roberta_base/")


#creating dictionary of token and index of word2vec after training
word2vec_target_dict = {}
# (word_embeddings): Embedding(50265, 768, padding_idx=1) 
source_matrix_emb = source_model.get_input_embeddings().weight.detach().numpy().copy()
target_matrix_emb = np.zeros(
        (len(target_to_source_dict), source_matrix_emb.shape[1]), dtype=source_matrix_emb.dtype
    )

matrix_size = target_matrix_emb.shape
print("Size of target_matrix_emb:", matrix_size)

cnt_index = 0
list_token = []
repeated_count = 0
for token, value in target_to_source_dict.items():
    if token in list_token:
        repeated_count+=1
        print("repeated count is: ", repeated_count)

    source_token = value['source_token']
    target_token_id = value['target_token_id']
    source_token_id = value['source_token_id']

    #initialing the initial target matrix embedding with the source matrix embedding
    word2vec_target_dict[token] = cnt_index
    # print("cnt_indx is: ", cnt_index)
    target_matrix_emb[cnt_index] = source_matrix_emb[source_token_id]  
    list_token.append(token)
    cnt_index += 1


target_vocab = target_tokenizer.get_vocab()
for index, token in enumerate(target_vocab):
    if token not in list_token:
        list_token.append(token)
        word2vec_target_dict[token] = cnt_index
        cnt_index += 1

###########verifying ###################
print(cnt_index)
print("length of tokenizer is: ", len(target_tokenizer))

embedding_size = 768
W_embeddings =   target_matrix_emb

class Word2VecDataset(object):
    def __init__(self, corpus, list_token, min_count=1, window_size=args.window_size, threshold=args.threshold):
        """ Prepares the training data for the word2vec neural network.
            Params:
                corpus (string): corpus of words
                min_count (int): words with minimum occurrence to consider
                window_size (int): context window size for generating word pairs
                threshold (float): threshold used for subsampling words
        """
        self.window_size = window_size
        self.min_count = min_count
        self.threshold = threshold
        print("self.threshold is: ", self.threshold)

        tokens = corpus.split()
        word_counts = Counter(tokens)
        word_counts = Counter({word:count for word, count in word_counts.items() if count >= min_count})        
        word_list = list_token
        self.word2idx = {word: idx for idx, word in enumerate(word_list)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        word_freq = np.zeros(len(self.word2idx))              #create count for every token
        for word, count in word_counts.items():
            if word in self.word2idx:  # Ensure the word is in the tokenizer's vocabulary
                idx = self.word2idx[word]
                word_freq[idx] = count

        # create prob dist based on word frequency
        self.unigram_dist = word_freq / word_freq.sum()
        # create prob dist for negative sampling
        self.noise_dist = self.unigram_dist ** 0.75
        self.noise_dist = self.noise_dist / self.noise_dist.sum()

        # get prob for drop words
        self.word_drop_prob = np.zeros(len(self.word2idx))
        for word, idx in self.word2idx.items():
            if word_freq[idx] > 0:  
                prob = 1 - np.sqrt(threshold / word_freq[idx])
                self.word_drop_prob[idx] = min(prob, 1.0) 
            else:
                self.word_drop_prob[idx] = 0  # Set drop probability to 0 for words not in corpus
                    
        self.generate_word_pairs()
    

    def generate_word_pairs(self):
        """ Creates the pairs of center and context words based on the context window size.
        """
        word_pair_ids = []
        lines = corpus.split('\n')
        for line in lines:
            tokens_l = line.split()
            token_ids = [self.word2idx.get(word, -1) for word in tokens_l]
            for current_idx, center_word_id in enumerate(token_ids):
                if center_word_id == -1:
                    continue
                
                if random.random() > self.word_drop_prob[center_word_id]:
                    left_boundary = max(current_idx - self.window_size, 0)
                    right_boundary = min(current_idx + self.window_size + 1, len(token_ids))
                    for context_position in range(left_boundary, right_boundary):
                        if context_position != current_idx: 
                            context_word_id = token_ids[context_position]
                            if context_word_id == -1:
                                continue
                            word_pair_ids.append((center_word_id, context_word_id))
        
        
        len_word_pair = len(word_pair_ids)
        print("length of word pairs is: ", len(word_pair_ids))
        wandb.log({"len_word_pair": len_word_pair })
        
        self.word_pair_ids = word_pair_ids

    def get_batches(self, batch_size):
        """ Creates the batches for training the network.
            Params:
                batch_size (int): size of the batch
            Returns:
                batch (torch tensor of shape (batch_size, 2)): tensor of word pair ids for a given batch
        """
        for i in range(0, len(self.word_pair_ids), batch_size):
            yield torch.tensor(self.word_pair_ids[i: i+batch_size], dtype=torch.long)
    
    
    def get_negative_samples(self, batch_size, n_samples):
        """ Samples negative word ids for a given batch.
            Params:
                batch_size (int): size of the batch
                n_samples (int): number of negative samples
            Returns:
                neg_samples (torch tensor of shape (batch_size, n_samples)): tensor of negative sample word ids
                    for a given batch
        """
        neg_samples_ids = np.random.choice(len(self.word2idx), size=(batch_size, n_samples), 
                                       replace=False, p=self.noise_dist)
        return torch.tensor(neg_samples_ids, dtype=torch.long)



with open(f'/data/nandini/vocab_adapt/codes/BERT/word2vec_tokenized_dataset/{args.dataset_file}.txt', encoding='utf-8') as f:
    corpus = f.read()

dataset = Word2VecDataset(corpus, list_token)

# Word2Vec Parameter
vocab_size = len(list_token)
print("vocab size is: ", vocab_size, " len according to dataset is: ", len(dataset.word2idx))


# Model
class Word2Vec(nn.Module):
    def __init__(self, W_init):      #w_init, v_init is the embedding layer and lm head initialization matrix respectively
        super(Word2Vec, self).__init__()
        self.total_size, self.embedding_dim = W_init.shape
        self.original_size = 19986
        self.new_size = 37964
        self.factorized_dim = 1024
        
        self.W_original = nn.Parameter(torch.tensor(W_init), requires_grad=False)  #torch.Size([32000, 4096])
        
        # Learnable transformation matrices, now factorized to a lower dimension
        self.A_W_1 = nn.Parameter(torch.rand(self.new_size, self.factorized_dim), requires_grad=True)  # torch.Size([37964, 1024])
        self.A_W_2 = nn.Parameter(torch.rand(self.factorized_dim, self.original_size), requires_grad=True)  # torch.Size([1024, 19984])

        self.combined_W = None
     


    def forward(self, in_ids, pos_out_ids, neg_out_ids):
        
        A_W = torch.matmul(self.A_W_1, self.A_W_2)
        
        # F.softmax(self.A_W, dim=-1) represents the probability score, and W_original represents the original embedding. 
        # Thus, the result is a linear combination of the original embedding, weighted by the probability score.
        # Here, new_embedding is actually the embedding of the newly added token, with 31,199 new tokens added.
        new_embeddings_W = F.softmax(A_W, dim=-1).mm(self.W_original) 
       
        # We are concatenating the final new embedding, which is the original embedding plus the embedding of the new token. 
        # This combined_W represents the word embedding layer of the final model.
        self.combined_W = torch.cat((self.W_original, new_embeddings_W), dim=0) 
       
        
        emb_in = self.combined_W[in_ids]
        pos_emb_out = self.combined_W[pos_out_ids]
        neg_emb_out = self.combined_W[neg_out_ids]

        # calculate loss for true pair
        # ----------------------------
        # step 1 is calculate the dot product between the input and output word embeddings
        pos_loss = torch.mul(pos_emb_out, emb_in)      # element-wise multiplication
        pos_loss = torch.sum(pos_loss, dim=1)           # sum the element-wise components
        
        # step 2 is to calculate the log sogmoid of dot product
        pos_loss = -F.logsigmoid(pos_loss)

        # calculate loss for negative pairs
        # ----------------------------------
        # step 1 is calculate the dot product between the input and output word embeddings
        neg_loss = torch.bmm(neg_emb_out, emb_in.unsqueeze(2)).squeeze()   # matrix-matrix multiplication
        neg_loss = torch.sum(neg_loss, dim=1)                               # sum the element-wise components

        # step 2 is to calculate the log sogmoid of dot product
        neg_loss = -F.logsigmoid(-neg_loss)

        return torch.mean(pos_loss + neg_loss)
        

scaler = GradScaler()
model = Word2Vec(W_embeddings).to(device)

print(model)

for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}, grad_fn={param.grad_fn}")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

# Total number of trainable parameters: 59338752
# Total number of non-trainable parameters: 15347712

print(f"Total number of trainable parameters: {trainable_params}")
print(f"Total number of non-trainable parameters: {non_trainable_params}")

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


n_epochs = args.epoch
n_neg_samples = 5
batch_size = args.batch_size



print("-" * 60)
print("Start of training")
print("-" * 60)

folder_name = f"weights_{args.dataset_file}_{args.threshold}_{args.batch_size}_{args.learning_rate}_{args.window_size}_{args.mlm_probability}"
base_path = "/data/nandini/vocab_adapt/codes/BERT/word2vec_weights_roberta/"
full_path = os.path.join(base_path, folder_name)
if not os.path.exists(full_path):
    os.makedirs(full_path)
    print(f"Folder '{full_path}' created successfully.")
else:
    print(f"Folder '{full_path}' already exists.")


for epoch in range(n_epochs):
    losses = []
    start = time.time()
    bt = 0

    for batch in dataset.get_batches(batch_size):
        # get the negative samples
        noise_word_ids = dataset.get_negative_samples(len(batch), n_neg_samples)

        # load tensor to GPU
        input_word_ids = batch[:, 0].to(device)
        target_word_ids = batch[:, 1].to(device)
        noise_word_ids = noise_word_ids.to(device)
        
        # forward pass
        loss = model.forward(input_word_ids, target_word_ids, noise_word_ids)

        # backward pass, optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        # del input_word_ids, target_word_ids, noise_word_ids
        # gc.collect()
        torch.cuda.empty_cache()

        
        if bt % 1000 == 0:
            print("in epoch: ", epoch, " batch is: ", bt, f" Avg training loss: {np.mean(losses):.6f}")
        
        bt += 1

    
    end = time.time()

    print(f"Epochs: {epoch}/{n_epochs}\tAvg training loss: {np.mean(losses):.6f}\tEllapsed time: {(end - start):.0f} s")
    wandb.log({"epoch-{}-loss".format(epoch): np.mean(losses)})
    wandb.log({
        "Elapsed time (s)": end - start
    })

    print("shape of combined_1 is: ", model.combined_W.shape)
    final_matrix = torch.zeros(model.combined_W.shape, dtype=model.combined_W.dtype)
    catwalk = 0
    for index, token in enumerate(target_vocab):
        final_matrix[index] = model.combined_W[word2vec_target_dict[token]]
        
        # if catwalk % 1000 == 0:
        #     print("index is: ", index, " token is: ", token)



    torch.save(final_matrix, f'/data/nandini/vocab_adapt/codes/BERT/word2vec_weights_roberta/{folder_name}/combined_W_epoch_{epoch}.pt')
    

print("-" * 60)
print("End of training")
print("-" * 60)
del model
gc.collect()

######################################### Initializing with the embedding and checking perplexity##########################
torch.cuda.empty_cache()

for epoch in range(0, n_epochs):
    emb_path =f'/data/nandini/vocab_adapt/codes/BERT/word2vec_weights_roberta/{folder_name}/combined_W_epoch_{epoch}.pt'
    target_matrix_emb = torch.load(emb_path)

    config = AutoConfig.from_pretrained(args.model_config, trust_remote_code=True)


    source_model = AutoModelForMaskedLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
    )

    
    
    config.vocab_size = len(target_matrix_emb)
    model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)

    new_bias = torch.zeros(len(target_tokenizer.get_vocab()))
    
    model.state_dict()['roberta.embeddings.word_embeddings.weight'].copy_(target_matrix_emb)   #word embedding setting 
    model.state_dict()['lm_head.decoder.bias'].copy_(new_bias)  
    model.state_dict()['lm_head.bias'].copy_(new_bias)  

    for param in source_model.state_dict():
        print(param)
        if "roberta.embeddings.word_embeddings.weight" in param:
            print("do nothing ", param )
        elif "lm_head.decoder.weight" in param:
            print("do nothing ", param )
        elif ("lm_head.decoder.bias") in param:
            print("do nothing ", param )
        elif "lm_head.bias" in param:
            print("do nothing ", param)
        else :
            # print(param)
            model.state_dict()[param].copy_(source_model.state_dict()[param])
    
    
    # device = torch.device("cuda:1")
    model.to(device)

    ############################################3calculating perplexity #######################
    lang_data = ["english" , "hindi", "tamil", "german", "russian"]
    tokenizer = AutoTokenizer.from_pretrained('/data/nandini/vocab_adapt/codes/BERT/extended_llama_roberta/', use_fast = False)

    for lang in lang_data:
        train_test_split = load_from_disk(f"/data-3/nandini/hf_dataset_split/{lang}")
        test_data = train_test_split['test']
        print("dataset is : ", lang)
        print(test_data)

        def preprocess_function(examples):
            return tokenizer([" ".join(x) for x in examples["text"]])


        tokenized_data = test_data.map(
            preprocess_function,
            batched=True,
            num_proc=8,
            remove_columns=test_data.column_names,
        )

        block_size = 480


        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of block_size.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            return result

        lm_dataset = tokenized_data.map(group_texts, batched=True, num_proc=8)
        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)
        print("lm dataset is: ", lm_dataset)

        trainer = Trainer(
            model=model,
            eval_dataset=lm_dataset,
            data_collator=data_collator,
        )
        eval_results = trainer.evaluate()
        perplexity = math.exp(eval_results['eval_loss'])
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        wandb.log({"perp-{}-epoch-{}".format(lang, epoch): perplexity})

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaTokenizer
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
import torch
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import random
from collections import Counter


accelerator = Accelerator()
# import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")
# dtype = torch.float

#to create vocabulary and the input data
source_tokenizer = AutoTokenizer.from_pretrained('/data/nandini/vocab_adapt/codes/llama_tokenizer')
target_tokenizer = AutoTokenizer.from_pretrained('/data/nandini/vocab_adapt/codes/indic_llama_hat_m_filter/', use_fast = False)

#creates list of token
target_vocab = target_tokenizer.get_vocab()
list_token = []
for index, token in enumerate(target_vocab):
    list_token.append(token)


config = AutoConfig.from_pretrained("/data/nandini/vocab_adapt/codes/config_llama2/", trust_remote_code=True)
source_model = AutoModelForCausalLM.from_pretrained(
    "/data/nandini/vocab_adapt/codes/model_llama2/",
    config=config,
)

embedding_size = 4096  

# embedding_size = source_model.get_input_embeddings().weight.shape[0]
source_matrix_emb = source_model.get_input_embeddings().weight.detach().numpy().copy()
source_matrix_lmhead = source_model.state_dict()['lm_head.weight'].numpy()
source_vocab = source_tokenizer.get_vocab()
target_vocab = target_tokenizer.get_vocab()
print("source_ebed matrix size id: ", source_matrix_emb.shape)
print("embedding size is: ", embedding_size)


W_embeddings = source_matrix_emb
V_embeddings = source_matrix_lmhead

del source_model
torch.cuda.empty_cache()



class Word2VecDataset(object):
    def __init__(self, corpus, list_token, min_count=1, window_size=5, threshold=3):
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

        tokens = corpus.split()
        word_counts = Counter(tokens)
        word_counts = Counter({word:count for word, count in word_counts.items() if count >= min_count})        
        word_list = list_token
        self.word2idx = {word: idx for idx, word in enumerate(word_list)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        word_freq = np.zeros(len(self.word2idx))
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
        
        print("length of word pairs is: ", len(word_pair_ids))
        
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



with open(f'/data/nandini/vocab_adapt/codes/tokenized_dataset/code_mix_data_tokenization_4M.txt', encoding='utf-8') as f:
    corpus = f.read()

dataset = Word2VecDataset(corpus, list_token)


# Word2Vec Parameter
vocab_size = len(list_token)
print("vocab size is: ", vocab_size, " len according to dataset is: ", len(dataset.word2idx))


# Model
class Word2Vec(nn.Module):
    def __init__(self, W_init, V_init):      #w_init, v_init is the embedding layer and lm head initialization matrix respectively
        super(Word2Vec, self).__init__()
        self.total_size, self.embedding_dim = W_init.shape
        self.original_size = 32000
        self.new_size = 31199
        self.factorized_dim = 1024
        
        self.W_original = nn.Parameter(torch.tensor(W_init), requires_grad=False)  #torch.Size([32000, 4096])
        self.V_original = nn.Parameter(torch.tensor(V_init), requires_grad=False) #torch.Size([32000, 4096])

        # Learnable transformation matrices, now factorized to a lower dimension
        self.A_W_1 = nn.Parameter(torch.rand(self.new_size, self.factorized_dim), requires_grad=True)  # torch.Size([31199, 1024])
        self.A_W_2 = nn.Parameter(torch.rand(self.factorized_dim, self.original_size), requires_grad=True)  # torch.Size([1024, 32000])

        self.A_V_1 = nn.Parameter(torch.rand(self.new_size, self.factorized_dim), requires_grad=True)  # torch.Size([31199, 1024])
        self.A_V_2 = nn.Parameter(torch.rand(self.factorized_dim, self.original_size), requires_grad=True)  # torch.Size([1024, 32000]) 

        self.combined_W = None
        self.combined_V = None


    def forward(self, in_ids, pos_out_ids, neg_out_ids):
        
        A_W = torch.matmul(self.A_W_1, self.A_W_2)
        A_V = torch.matmul(self.A_V_1, self.A_V_2)
        # F.softmax(self.A_W, dim=-1) represents the probability score, and W_original represents the original embedding. 
        # Thus, the result is a linear combination of the original embedding, weighted by the probability score.
        # Here, new_embedding is actually the embedding of the newly added token, with 31,199 new tokens added.
        new_embeddings_W = F.softmax(A_W, dim=-1).mm(self.W_original)  #torch.Size([31199, 4096])
        new_embeddings_V = F.softmax(A_V, dim=-1).mm(self.V_original)  #torch.Size([31199, 4096])
        # We are concatenating the final new embedding, which is the original embedding plus the embedding of the new token. 
        # This combined_W represents the word embedding layer of the final model.
        self.combined_W = torch.cat((self.W_original, new_embeddings_W), dim=0)  #torch.Size([63199, 4096])
        self.combined_V = torch.cat((self.V_original, new_embeddings_V), dim=0)  #torch.Size([63199, 4096])
        
        emb_in = self.combined_W[in_ids]
        pos_emb_out = self.combined_V[pos_out_ids]
        neg_emb_out = self.combined_V[neg_out_ids]

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
model = Word2Vec(W_embeddings, V_embeddings).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0003)

n_epochs = 5
n_neg_samples = 5
batch_size = 1024

# Total number of trainable parameters: 129431552
# Total number of non-trainable parameters: 262144000

print("-" * 60)
print("Start of training")
print("-" * 60)

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

        
        if bt % 1000 == 0:
            print("in epoch: ", epoch, " batch is: ", bt, f" Avg training loss: {np.mean(losses):.6f}")
        
        bt += 1

    
    end = time.time()

    print(f"Epochs: {epoch + 1}/{n_epochs}\tAvg training loss: {np.mean(losses):.6f}\tEllapsed time: {(end - start):.0f} s")
    torch.save(model.combined_W, f'/data/nandini/vocab_adapt/codes/word2vec_weights/combined_W_epoch_{epoch}_1_03_3.pt')
    torch.save(model.combined_V, f'/data/nandini/vocab_adapt/codes/word2vec_weights/combined_V_epoch_{epoch}_1_03_3.pt')

print("-" * 60)
print("End of training")
print("-" * 60)




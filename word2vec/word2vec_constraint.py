import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaTokenizer
import torch.nn.functional as F
# import matplotlib.pyplot as plt

dtype = torch.FloatTensor

#to create vocabulary and the input data
source_tokenizer = AutoTokenizer.from_pretrained('/data-3/nandini/vocab_adapt/codes/llama_tokenizer')
target_tokenizer = AutoTokenizer.from_pretrained('/data-3/nandini/vocab_adapt/codes/indic_llama_hat_m_filter/', use_fast = False)

target_vocab = target_tokenizer.get_vocab()
list_token = []
for index, token in enumerate(target_vocab):
    list_token.append(token)

#passing the tokenized dataset
sentences = []
with open(f'/data-3/nandini/vocab_adapt/codes/tok_eng.txt', encoding='utf-8') as f:
    lines = f.readlines()
    sentences.extend(lines)


# list all the words present in our corpus
word_sequence = " ".join(sentences).split()
word_list = list_token
word_dict = {w: i for i, w in enumerate(word_list)}  #create dictionary word:index


# Word2Vec Parameter
batch_size = 20  
embedding_size = 4096  
voc_size = len(word_list)

# defined the context
context_size = 1

skip_grams = []
for i in range(1, len(word_sequence) - 1):
    input = word_dict[word_sequence[i]]  #index of word
    context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]     #index of neighboring word

    for w in context:
        skip_grams.append([input, w])      #skipgram append input , predicted


# np.random.seed(172)

#to initialize the matrix

config = AutoConfig.from_pretrained("/data-3/nandini/vocab_adapt/codes/config_llama2/", trust_remote_code=True)
source_model = AutoModelForCausalLM.from_pretrained(
    "/data-3/nandini/vocab_adapt/codes/model_llama2/",
    config=config,
)


# embedding_size = source_model.get_input_embeddings().weight.shape[0]
source_matrix_emb = source_model.get_input_embeddings().weight.detach().numpy().copy()
source_matrix_lmhead = source_model.state_dict()['lm_head.weight'].numpy()
source_vocab = source_tokenizer.get_vocab()
target_vocab = target_tokenizer.get_vocab()
print("source_ebed matrix size id: ", source_matrix_emb.shape)
print("embedding size is: ", embedding_size)

mean_emb, std_emb = (
            source_matrix_emb.mean(0),
            source_matrix_emb.std(0),
        )

mean_lmhead, std_lmhead = (
        source_matrix_lmhead.mean(0),
        source_matrix_lmhead.std(0),
    )


random_fallback_matrix_emb = np.random.RandomState(1234).normal(
        mean_emb, std_emb, (len(target_tokenizer.get_vocab()), source_matrix_emb.shape[1])
    )

random_fallback_matrix_lmhead = np.random.RandomState(1234).normal(
        mean_lmhead, std_lmhead, (len(target_tokenizer.get_vocab()), source_matrix_lmhead.shape[1])
    )

W_embeddings = np.zeros((voc_size, embedding_size))
V_embeddings = np.zeros((voc_size, embedding_size))


for index, token in enumerate(target_vocab):
    if token in source_vocab:
        W_embeddings[word_dict[token]] = source_matrix_emb[source_vocab[token]]
    else :
        W_embeddings[word_dict[token]] = random_fallback_matrix_emb[index]
    
    if token in source_vocab:
        V_embeddings[word_dict[token]] = source_matrix_lmhead[source_vocab[token]]
    else :
        V_embeddings[word_dict[token]] = random_fallback_matrix_lmhead[index]


def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)
    # print("random index is: ", random_index)

    for i in random_index:
        # one-hot encoding of words
        random_inputs.append(np.eye(voc_size)[data[i][0]])  # input  vocab (1X vocab_size)
        random_labels.append(data[i][1])  # context word
    

    return random_inputs, random_labels



# inputs: like , i, dog , context: i, dog, i


# Model
class Word2Vec(nn.Module):
    def __init__(self, W_init, V_init):      #w_init, v_init is the embedding layer and lm head initialization matrix respectively
        super(Word2Vec, self).__init__()
        self.total_size, self.embedding_dim = W_init.shape
        self.original_size = 32000
        self.new_size = 31199
        # parameters between -1 and + 1
        self.W = nn.Parameter(torch.tensor(W_init).float(), requires_grad=False)  # Convert numpy array to tensor W 

        self.V = nn.Parameter(torch.tensor(V_init).float().t(), requires_grad=False)  # Convert numpy array to tensor

        self.A_W = nn.Parameter(torch.rand(self.new_size, self.original_size))
        self.A_V = nn.Parameter(torch.rand(self.new_size, self.original_size))

        

    def forward(self, X):
        X = X.float()
        # Compute new embeddings as a combination of the original ones
        new_embeddings_W = F.softmax(self.A_W, dim=-1).mm(self.W[:self.original_size])
        new_embeddings_V = F.softmax(self.A_V, dim=-1).mm(self.V[:, :self.original_size])

        # Update the latter half of W and V with new embeddings indirectly through A_W and A_V
        with torch.no_grad():  
            self.W[self.original_size:] = new_embeddings_W
            self.V[:, self.original_size:] = new_embeddings_V.t()
        
        hidden_layer = torch.matmul(X, self.W)  # hidden_layer : [batch_size, embedding_size]
        output_layer = torch.matmul(hidden_layer, self.V)  # output_layer : [batch_size, voc_size]
        #return output_layer 
        return output_layer

model = Word2Vec(W_embeddings, V_embeddings)
# Set the model in train mode
model.train()

criterion = nn.CrossEntropyLoss() # Softmax (for multi-class classification problems) is already included
optimizer = optim.Adam(model.parameters(), lr=0.001)

# frozen_tokens = []
# tokens_to_freeze = []
# for index, token in enumerate(source_vocab):
#     frozen_tokens.append(token)
#     tokens_to_freeze.append(word_dict[token])

# W_mask = torch.ones_like(model.W)
# V_mask = torch.ones_like(model.V)
# for index in tokens_to_freeze:
#     W_mask[index] = 0
#     V_mask[:, index] = 0  #As it is transposed


# Training
for epoch in range(1):
    total_loss = 0
    # Loop through the entire dataset in batches
    for i in range(0, len(skip_grams), batch_size):
        # Generate a batch of data
        input_batch, target_batch = random_batch(skip_grams[i:i+batch_size], batch_size)

        # Convert to tensors
        input_batch = torch.tensor(np.array(input_batch))
        target_batch = torch.tensor(np.array(target_batch, dtype=np.int64))

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(input_batch)

        # Compute loss
        loss = criterion(output, target_batch)
        total_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        # model.W.grad *= W_mask
        # model.V.grad *= V_mask
        optimizer.step()
        print("after 1 batch")

    # Print average loss per epoch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_loss / (len(skip_grams) // batch_size)))

    
# Learned W
W, _= model.parameters()
print(W.detach())

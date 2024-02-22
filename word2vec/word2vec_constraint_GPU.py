import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaTokenizer
import torch.nn.functional as F
from accelerate import Accelerator

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

#passing the tokenized dataset
sentences = []
with open(f'/data/nandini/vocab_adapt/codes/tokenized_dataset/code_mix_data_tokenization_40M.txt', encoding='utf-8') as f:
    lines = f.readlines()
    sentences.extend(lines)


# list all the words present in our corpus
word_sequence = " ".join(sentences).split()

word_list = list_token
word_dict = {w: i for i, w in enumerate(word_list)}  #create dictionary word:index

####sanity check that every token correspond to the same tokenid in the matrix
for token, token_id in target_tokenizer.get_vocab().items():
    if token_id != word_dict[token]:
        print(token)

print("there is no such token")


# Word2Vec Parameter
batch_size = 8
embedding_size = 4096  
voc_size = len(word_list)

# defined the context
context_size = 1

skip_grams = []
for i in range(1, len(word_sequence) - 1):
    input = word_dict[word_sequence[i]]  #index of word
    #if want to change context size then this below line need to be changed
    context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]     #index of neighboring word

    for w in context:
        skip_grams.append([input, w])      #skipgram append input , predicted


# np.random.seed(172)

#to initialize the matrix

config = AutoConfig.from_pretrained("/data/nandini/vocab_adapt/codes/config_llama2/", trust_remote_code=True)
source_model = AutoModelForCausalLM.from_pretrained(
    "/data/nandini/vocab_adapt/codes/model_llama2/",
    config=config,
)


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



def random_batch(data, size):    #size is batch size, data is list of input and target word list
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)   #to shuffle the input

    for i in random_index:
        # one-hot encoding of words
        random_inputs.append(np.eye(voc_size)[data[i][0]])  # input  vocab (1X vocab_size) one-hot vector
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


    def forward(self, X):
        X = X.half()    #torch.Size([16, 63199]) batch_size = 16

        #multiplying the factorized
        # A_W = self.A_W_1.mm(self.A_W_2)
        # A_V = self.A_V_1.mm(self.A_V_2)

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

        
        # Compute the hidden layer and output using the combined embeddings. 
        # The output is the embedding from the word embedding layer
        hidden_layer = torch.matmul(X, self.combined_W)      #torch.Size([20, 4096])

        #output of lmhead layer
        output_layer = torch.matmul(hidden_layer, self.combined_V.t())       #torch.Size([20, 63199])
        return output_layer

model = Word2Vec(W_embeddings, V_embeddings)
model = model.to(device).half()



# Set the model in train mode
model.train()

#sanity check
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}, grad_fn={param.grad_fn}")

criterion = nn.CrossEntropyLoss() # Softmax (for multi-class classification problems) is already included
optimizer = optim.Adam(model.parameters(), lr=0.001)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

# Total number of trainable parameters: 1996736000
# Total number of non-trainable parameters: 262144000

print(f"Total number of trainable parameters: {trainable_params}")
print(f"Total number of non-trainable parameters: {non_trainable_params}")

for name, p in model.named_parameters():
    if p.requires_grad:
        print(f"{name}: shape={p.shape}")


# Training
for epoch in range(3):
    total_loss = 0
    # Loop through the entire dataset in batches
    for i in range(0, len(skip_grams), batch_size):
        if i % 10000 == 0:
            print("in nested for loop i is: ", i)
        
        # Generate a batch of data
        input_batch, target_batch = random_batch(skip_grams[i:i+batch_size], batch_size)
       

        # Convert to tensors
        input_batch = torch.tensor(np.array(input_batch)).to(device).half()      #torch.Size([16, 63199])
        target_batch = torch.tensor(np.array(target_batch, dtype=np.int64)).to(device)      #torch.Size([16])
  
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        output = model(input_batch)
        # _, predicted_classes = torch.max(output, 1)
        # print("predicted class is::::::::::: ")
        # print(predicted_classes)
        
        # Compute loss
        loss = criterion(output, target_batch)
        # print("Loss grad_fn:", loss.grad_fn)
        # loss.requires_grad = True

        total_loss += loss.item()
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    #to save final embedding after each layer
    torch.save(model.combined_W, f'/data/nandini/vocab_adapt/codes/word2vec_weights/combined_W_epoch_{epoch}_1.pt')
    torch.save(model.combined_V, f'/data/nandini/vocab_adapt/codes/word2vec_weights/combined_V_epoch_{epoch}_1.pt')
    # Print average loss per epoch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_loss / (len(skip_grams) // batch_size)))

    

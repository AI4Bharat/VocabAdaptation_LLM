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
source_tokenizer = AutoTokenizer.from_pretrained('/data/nandini/vocab_adapt/codes/llama_tokenizer')
target_tokenizer = AutoTokenizer.from_pretrained('/data/nandini/vocab_adapt/codes/indic_llama_hat_m_filter/', use_fast = False)

target_vocab = target_tokenizer.get_vocab()
list_token = []
for index, token in enumerate(target_vocab):
    list_token.append(token)

#passing the tokenized dataset
sentences = []
with open(f'/data/nandini/vocab_adapt/codes/tok_eng.txt', encoding='utf-8') as f:
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
        self.W_original = nn.Parameter(torch.tensor(W_init).float(), requires_grad=False)  #torch.Size([32000, 4096])
        self.V_original = nn.Parameter(torch.tensor(V_init).float(), requires_grad=False) #torch.Size([32000, 4096])

        # Learnable transformation matrices
        self.A_W = nn.Parameter(torch.rand(self.new_size, self.original_size), requires_grad=True)      #torch.Size([31199, 32000])
        self.A_V = nn.Parameter(torch.rand(self.new_size, self.original_size), requires_grad=True)      #torch.Size([31199, 32000]) 

        self.combined_W = None
        self.combined_V = None


    def forward(self, X):
        X = X.float()
        
        # Compute new embeddings dynamically
        new_embeddings_W = F.softmax(self.A_W, dim=-1).mm(self.W_original)  #torch.Size([31199, 4096])
        new_embeddings_V = F.softmax(self.A_V, dim=-1).mm(self.V_original)  #torch.Size([31199, 4096])

        print("AW is:::::")
        print(A_W)

        print("softmax of A_W is: ::::::::::::::")
        print(F.softmax(self.A_W, dim=-1))


        print("Shape of A_W:", self.A_W.shape)
        print("Shape of A_V:", self.A_V.shape)

        print("Shape of W_original:", self.W_original.shape)
        print("Shape of V_original:", self.V_original.shape)        

        print("Shape of new_embeddings_W:", new_embeddings_W.shape)
        print("Shape of new_embeddings_V:", new_embeddings_V.shape)

        # Use both original and new embeddings for computation
        self.combined_W = torch.cat((self.W_original, new_embeddings_W), dim=0)  #torch.Size([63199, 4096])
        self.combined_V = torch.cat((self.V_original, new_embeddings_V), dim=0)  #torch.Size([63199, 4096])

        print("Shape of combined_W:", self.combined_W.shape)
        print("Shape of combined_V:", self.combined_V.shape)
        
        # Compute the hidden layer and output using the combined embeddings
        hidden_layer = torch.matmul(X, self.combined_W)      #torch.Size([20, 4096])
        output_layer = torch.matmul(hidden_layer, self.combined_V.t())       #torch.Size([20, 63199])

        print("Shape of hidden_layer:", hidden_layer.shape)
        print("Shape of output_layer:", output_layer.shape)
        return output_layer

model = Word2Vec(W_embeddings, V_embeddings)
# Set the model in train mode

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)

model.train()
print(model)

for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}, grad_fn={param.grad_fn}")

criterion = nn.CrossEntropyLoss() # Softmax (for multi-class classification problems) is already included
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(1):
    total_loss = 0
    # Loop through the entire dataset in batches
    for i in range(0, len(skip_grams), batch_size):
        print("in nested for loop i is: ", i)
        # Generate a batch of data
        input_batch, target_batch = random_batch(skip_grams[i:i+batch_size], batch_size)

        # Convert to tensors
        input_batch = torch.tensor(np.array(input_batch))
        target_batch = torch.tensor(np.array(target_batch, dtype=np.int64))

        print("Shape of input_batch:", input_batch.shape)
        print("Shape of target_batch:", target_batch.shape)
        # Zero gradients
        optimizer.zero_grad()

        print("after optimizer.zero grad")
        # Forward pass
        output = model(input_batch)

        print("after output = model(input_batch)")

        # Compute loss
        loss = criterion(output, target_batch)
        print("Loss grad_fn:", loss.grad_fn)
        # loss.requires_grad = True

        print("after defining loss")
        total_loss += loss.item()

        print("after train_lkoss = loss.item()")

        # Backward pass and optimize
        loss.backward()

        print("after loss_backward()")
        optimizer.step()
        print("after 1 batch")

    torch.save(model.combined_W, f'combined_W_epoch_{epoch}.pt')
    torch.save(model.combined_V, f'combined_V_epoch_{epoch}.pt')

    # Print average loss per epoch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_loss / (len(skip_grams) // batch_size)))

    
# Learned W
W, _= model.parameters()
print(W.detach())

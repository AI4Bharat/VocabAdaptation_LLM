import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaTokenizer
# import matplotlib.pyplot as plt

dtype = torch.FloatTensor
source_tokenizer = AutoTokenizer.from_pretrained('/data-3/nandini/vocab_adapt/codes/llama_tokenizer')
target_tokenizer = AutoTokenizer.from_pretrained('/data-3/nandini/vocab_adapt/codes/indic_llama_hat_m_filter/', use_fast = False)

target_vocab = target_tokenizer.get_vocab()
list_token = []
for index, token in enumerate(target_vocab):
    list_token.append(token)

sentences = []
with open(f'/data-3/nandini/vocab_adapt/codes/tok_eng.txt', encoding='utf-8') as f:
    lines = f.readlines()
    sentences.extend(lines)


# list all the words present in our corpus
word_sequence = " ".join(sentences).split()

word_list = list_token

word_dict = {w: i for i, w in enumerate(word_list)}  #create dictionary word:index


# Word2Vec Parameter
batch_size = 20  # To show 2 dim embedding graph
embedding_size = 4096  # To show 2 dim embedding graph
voc_size = len(word_list)

# input word
j = 1
# print("Input word : ")
# print(word_sequence[j], word_dict[word_sequence[j]])

# # context words
# print("Context words : ")
# print(word_sequence[j - 1], word_sequence[j + 1])
# print([word_dict[word_sequence[j - 1]], word_dict[word_sequence[j + 1]]])

# Make skip gram of one size window
skip_grams = []
for i in range(1, len(word_sequence) - 1):
    input = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]

    for w in context:
        skip_grams.append([input, w])


#lets plot some data
skip_grams[:6]


np.random.seed(172)

def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        # one-hot encoding of words
        random_inputs.append(np.eye(voc_size)[data[i][0]])  # input
        random_labels.append(data[i][1])  # context word

    return random_inputs, random_labels

random_batch(skip_grams[:6], size=3)

# inputs: like , i, dog , context: i, dog, i


# Model
class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()

        # parameters between -1 and + 1
        self.W = nn.Parameter(-2 * torch.rand(voc_size, embedding_size) + 1).type(dtype) # voc_size -> embedding_size Weight
        self.V = nn.Parameter(-2 * torch.rand(embedding_size, voc_size) + 1).type(dtype) # embedding_size -> voc_size Weight

    def forward(self, X):
        X = X.float()
        hidden_layer = torch.matmul(X, self.W) # hidden_layer : [batch_size, embedding_size]
        output_layer = torch.matmul(hidden_layer, self.V) # output_layer : [batch_size, voc_size]
        #return output_layer 
        return output_layer

model = Word2Vec()
# Set the model in train mode
model.train()

criterion = nn.CrossEntropyLoss() # Softmax (for multi-class classification problems) is already included
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(1):
    total_loss = 0
    # Loop through the entire dataset in batches
    for i in range(0, len(skip_grams), batch_size):
        # Generate a batch of data
        input_batch, target_batch = random_batch(skip_grams[i:i+batch_size], batch_size)

        # Convert to tensors
        input_batch = torch.tensor(np.array(input_batch))
        # input_batch = torch.tensor(input_batch)
        target_batch = torch.tensor(np.array(target_batch, dtype=np.int64))
        # target_batch = torch.tensor(target_batch)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(input_batch)

        # Compute loss
        loss = criterion(output, target_batch)
        total_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        # print("after 1 batch")

    # Print average loss per epoch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_loss / (len(skip_grams) // batch_size)))

    
# Learned W
W, _= model.parameters()
print(W.detach())

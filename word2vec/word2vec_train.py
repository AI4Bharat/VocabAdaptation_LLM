from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaTokenizer
import numpy as np

source_tokenizer = AutoTokenizer.from_pretrained('/data-3/nandini/vocab_adapt/codes/llama_tokenizer')
target_tokenizer = AutoTokenizer.from_pretrained('/data-3/nandini/vocab_adapt/codes/indic_llama_hat_m_filter/', use_fast = False)

target_vocab = target_tokenizer.get_vocab()
list_token = []
for index, token in enumerate(target_vocab):
    list_token.append(token)

print(len(list_token))

# print(target_vocab)
model = Word2Vec(vector_size=4096, min_count=1)
model.build_vocab([list_token])

config = AutoConfig.from_pretrained("/data-3/nandini/vocab_adapt/codes/config_llama2/", trust_remote_code=True)
source_model = AutoModelForCausalLM.from_pretrained(
    "/data-3/nandini/vocab_adapt/codes/model_llama2/",
    config=config,
)


embedding_size = source_model.get_input_embeddings().weight.shape[0]
source_matrix_emb = source_model.get_input_embeddings().weight.detach().numpy().copy()
source_matrix_lmhead = source_model.state_dict()['lm_head.weight'].numpy()
source_vocab = source_tokenizer.get_vocab()
target_vocab = target_tokenizer.get_vocab()

target_matrix_emb = np.zeros(
        (len(target_tokenizer), source_matrix_emb.shape[1]), dtype=source_matrix_emb.dtype
    )

target_matrix_lmhead = np.zeros(
        (len(target_tokenizer), source_matrix_lmhead.shape[1]), dtype=source_matrix_lmhead.dtype
    )



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

for index, token in enumerate(target_vocab):
    if token in source_vocab:
        model.wv.vectors[model.wv.get_index(token)] = source_matrix_emb[source_vocab[token]]
    else :
        model.wv.vectors[model.wv.get_index(token)] = random_fallback_matrix_emb[index]
    
    if token in source_vocab:
        model.syn1neg[model.wv.get_index(token)] = source_matrix_lmhead[source_vocab[token]]
    else :
        model.syn1neg[model.wv.get_index(token)] = random_fallback_matrix_lmhead[index]



frozen_tokens = []
for index, token in enumerate(source_vocab):
    frozen_tokens.append(token)

original_input_vectors = {token: model.wv[token] for token in frozen_tokens if token in model.wv}
original_output_vectors = {token: model.syn1neg[model.wv.get_index(token)] for token in frozen_tokens if token in model.wv}

class FreezeWeightsCallback(CallbackAny2Vec):
    def __init__(self, original_input_vectors, original_output_vectors):
        self.original_input_vectors = original_input_vectors
        self.original_output_vectors = original_output_vectors

    def on_epoch_end(self, model):
        for token, vector in self.original_input_vectors.items():
            if token in model.wv:
                model.wv[token] = vector
        for token, vector in self.original_output_vectors.items():
            if token in model.wv:
                model.syn1neg[model.wv.get_index(token)] = vector



callback = FreezeWeightsCallback(original_input_vectors, original_output_vectors)
sentences = [
    ["cat", "sat", "on", "the", "mat"],
    ["dog", "sat", "on", "the", "rug"],
    ["cat", "chased", "the", "dog"]
]
print("training started")
model.train(sentences, total_examples=model.corpus_count, epochs=5, callbacks=[callback])


# model.train()





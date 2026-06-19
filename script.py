import torch
import torch.nn as nn
from torch.nn import functional as F

with open("input.txt", "r", encoding='utf-8') as handle:
    text = handle.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)

stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# print(encode("hello World"))
# print(decode(encode("hello World")))

data = torch.tensor(encode(text), dtype=torch.long)

# print(data.shape, data.dtype)
# print(data[:1000])

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = tprch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, target):
        logits = self.token_embedding_table(idx)
        return logits

m = BigramLanguageModel(vocab_size)
out = m(xb, yb) 
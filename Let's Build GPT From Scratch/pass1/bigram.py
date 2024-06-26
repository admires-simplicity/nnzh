import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all unique characters in text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# mapping from chars to ints
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train/validate splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 90% train
train_data = data[:n]
val_data = data[n:]

# batch loading
def get_batch(split='test'):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    
# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        # idx and targets are each tensors with shape (B, T)
        logits = self.token_embedding_table(idx) # (B,T,C) "batch, time, channel" -- really "time" is position in the input block
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, _ = self(idx) # get predictions for each context in batch. (B, T, C)
            logits = logits[:, -1, :] # get prediction for last position (last char) in each context.
            # logits collapses into (B, C). I think this is because T becomes 1, and the fact that
            # it collapses into this instead of (B, 1, C) is just pytorch behaviour.
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train loop
for iter in range(max_iters):

    # periodically eval and report loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'step {iter}, train loss: {losses["train"]:.4f}, val loss: {losses["val"]:.4f}')

    # sample a batch
    xb, yb = get_batch('train')

    # eval loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from model
ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(ctx, max_new_tokens=500)[0].tolist()))
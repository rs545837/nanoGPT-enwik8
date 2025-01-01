# Quick Start

### Start by downloading all the requirements from requirementst.txt:
```python 
pip install -r requirements.txt
```

### Put the preprocess.py file into the data directory of the repository and then run it:
```python
python preprocess.py
```

### Create a train_enwik8_char.py for the training config and model.py and use the ones in this repository, and start the training using:
```python
python train.py config/train_enwik8_char.py
```

### Evaluate the checkpoint but before that copy and paste the evaluate.py file into the main nanoGPT repo.
```python
python evaluate.py --checkpoint out-enwik8-char/ckpt.pt
```



Now modify the CausalSelfAttention class to use rotary embeddings:

```      
pip install rotary-embedding-torch
```
      
```python
from rotary_embedding_torch import RotaryEmbedding

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Initialize the rotary embedding
        self.rotary_emb = RotaryEmbedding(dim=self.n_embd // self.n_head)

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Flash attention or manual implementation (PyTorch >= 2.0 supports flash)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimensionality (n_embd)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape and apply rotary embeddings to the query and key
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Apply rotary embeddings
        q = self.rotary_emb(q)
        k = self.rotary_emb(k)

        # Perform attention (flash or manual)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))
        return y
```
This modification adds rotary positional embeddings to your self-attention layer, which helps your model better handle long sequences, especially at character-level granularity.

2. Parameter-Efficient Fine-Tuning (LoRA)
LoRA (Low-Rank Adaptation) allows you to train only a small number of additional parameters, while the base model remains frozen, which makes training more efficient.

Code Change (LoRA Integration)
You can modify the CausalSelfAttention class to integrate LoRA by inserting low-rank adaptation layers into the query and key projections.

Install peft (parameter-efficient fine-tuning) with:

```bash
pip install peft
```
Then, modify your CausalSelfAttention class:

```python
from peft import LoraLinear

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Replace standard Linear layers with LoRA layers for q and k
        self.q_proj = LoraLinear(config.n_embd, config.n_embd, r=4, lora_alpha=16, dropout=config.dropout)
        self.k_proj = LoraLinear(config.n_embd, config.n_embd, r=4, lora_alpha=16, dropout=config.dropout)

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Apply LoRA to q and k projections
        q = self.q_proj(q)
        k = self.k_proj(k)

        # Continue with attention mechanism...
        # (same as the rest of the forward pass from your previous attention implementation)
```
This addition of LoRA layers in the query (q) and key (k) projections makes your model more efficient in fine-tuning without modifying the base model weights.

3. Layer-wise Learning Rate Decay
We already suggested a dynamic learning rate decay mechanism. Here’s the full implementation in the configure_optimizers function:

(Layer-wise Learning Rate Decay)
```python
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    decay_params = []
    nodecay_params = []
    lr_decay_params = []
    lr_no_decay_params = []

    for layer_idx, (n, p) in enumerate(self.named_parameters()):
        if p.requires_grad:
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)
            # Apply layer-wise learning rate decay
            lr_decay_params.append({
                'params': p,
                'lr': learning_rate * (0.95 ** layer_idx),  # decay per layer
                'weight_decay': weight_decay
            })
            lr_no_decay_params.append({
                'params': p,
                'lr': learning_rate,
                'weight_decay': 0.0
            })

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ], lr=learning_rate, betas=betas)
    
    # Optionally add layer-wise decay groups (used for fine-tuning)
    lr_decay_optimizer = torch.optim.AdamW(lr_decay_params, betas=betas)

    return optimizer, lr_decay_optimizer
```
This code sets up both standard parameter decay and layer-wise learning rate decay.

4. Efficient Evaluation using Flash Attention
Flash attention can be used for efficient evaluation if you are using PyTorch 2.0 or higher. You already have a check for this in your code, but make sure it's being utilized.

Code Change (Using Flash Attention for Evaluation)

```python
def forward(self, x):
    B, T, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

    # Check for Flash attention in evaluation mode
    if not self.training and self.flash:
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    else:
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

    return y
```



> step 2000: train loss 2.0577, train perplexity 7.8959, val loss 2.0154, val perplexity 7.5307
> saving checkpoint to out-enwik8-char
> iter 2000: loss 1.9532, time 4379.76ms, mfu 0.01%


> iter 4990: loss 1.2068, time 31.28ms, mfu 14.28%
> step 5000: train loss 1.1379, train perplexity 3.1222, val loss 1.1532, val perplexity 3.1699
> saving checkpoint to out-enwik8-char
> iter 5000: loss 1.2180, time 5242.55ms, mfu 12.86%

> step 5000: train loss 1.1375, train perplexity 3.1209, val loss 1.1526, val perplexity 3.1680
> saving checkpoint to out-enwik8-char
> iter 5000: loss 1.2152, time 4920.90ms, mfu 12.61%


> step 5000: train loss 1.1395, train perplexity 3.1270, val loss 1.1548, val perplexity 3.1750
> saving checkpoint to out-enwik8-char
> iter 5000: loss 1.2174, time 4907.41ms, mfu 12.61%

> step 5000: train loss 1.2984, train perplexity 3.6654, val loss 1.3087, val perplexity 3.7034
> saving checkpoint to out-enwik8-char
> iter 5000: loss 1.3470, time 7913.22ms, mfu 10.40%



> step 10000: train loss 0.9650, train perplexity 2.6254, val loss 1.0029, val perplexity 2.7269
> iter 10000: loss 0.9926, time 16826.37ms, mfu 12.81%


> [!NOTE]  
> with model_v1.py
> step 5000: train loss 1.1115, train perplexity 3.0409, val loss 1.1284, val perplexity 3.0921
> saving checkpoint to out-enwik8-char
> iter 5000: loss 1.1900, time 6163.89ms, mfu 11.48%

> [!NOTE]
> with model_v2.py
> step 5000: train loss 1.1141, train perplexity 3.0485, val loss 1.1358, val perplexity 3.1156
> saving checkpoint to out-enwik8-char
> iter 5000: loss 1.1838, time 6324.16ms, mfu 11.69%

> [!NOTE]
> with model_v3.py
> step 5000: train loss 2.0658, train perplexity 7.8995, val loss 2.0723, val perplexity 7.9497
> iter 5000: loss 1.4104, time 6126.37ms, mfu 10.34%

> [!NOTE]
> with model_v4.py
> step 5000: train loss 1.1218, train perplexity 3.0721, val loss 1.1430, val perplexity 3.1379
> saving checkpoint to out-enwik8-char
> iter 5000: loss 1.1856, time 5819.57ms, mfu 11.75%


> [!NOTE]
> learning rate: 1e-3
> > with model_v4.py
> step 10000: train loss 1.0556, train perplexity 2.8750, val loss 1.0791, val perplexity 2.9437
> iter 10000: loss 1.1129, time 5650.78ms, mfu 11.88%

> [!NOTE]
> learning rate: 2e-3
> > with model_v4.py
step 10000: train loss 1.0588, train perplexity 2.8845, val loss 1.0831, val perplexity 2.9556
iter 10000: loss 1.1160, time 5935.31ms, mfu 12.08%

> [!NOTE]
> learning rate: 2e-3
> > with model_v4.py
step 15000: train loss 1.0226, train perplexity 2.7818, val loss 1.0512, val perplexity 2.8625
iter 15000: loss 1.0434, time 5922.00ms, mfu 11.68%

> [!NOTE]
> learning rate: 2e-3
> > > with model_v5.py
step 15000: train loss 1.0266, train perplexity 2.7930, val loss 1.0499, val perplexity 2.8586
iter 15000: loss 1.0771, time 8009.14ms, mfu 8.73%

> [!NOTE]
> learning rate: 2e-3
> > > with model_v6.py
step 15000: train loss 0.9886, train perplexity 2.6886, val loss 1.0234, val perplexity 2.7839
iter 15000: loss 0.9749, time 8946.45ms, mfu 9.56%


> [!NOTE]
> learning rate: 2e-3
> > > with model_v7.py
step 15000: train loss 1.0571, train perplexity 2.8795, val loss 1.0878, val perplexity 2.9691
iter 15000: loss 1.0515, time 8519.65ms, mfu 9.28%



> [!NOTE]
> learning rate: 2e-3
> > > with model_v6.py
step 25000: train loss 0.9511, train perplexity 2.5901, val loss 0.9875, val perplexity 2.6859
saving checkpoint to out-enwik8-char
iter 25000: loss 0.9982, time 9437.11ms, mfu 9.18%


> [!NOTE]
> learning rate: 8e-4
> > > with model_v6.py, great_config.py
step 50000: train loss 0.7800, train perplexity 2.1822, val loss 0.8868, val perplexity 2.4284
iter 50000: loss 0.8363, time 19203.31ms, mfu 18.09%


> [!NOTE]
> learning rate: 1e-3
> > > with model_v6.py, great_config2.py
step 27000: train loss 0.6514, train perplexity 1.9185, val loss 0.8296, val perplexity 2.2928
iter 27000: loss 0.7029, time 132372.65ms, mfu 14.71%

> [!NOTE]
> learning rate: 1e-3
> > > with model_v6.py, great_config2.py
step 50000: train loss 0.5118, train perplexity 1.6685, val loss 0.8679, val perplexity 2.3825
iter 50000: loss 0.5606, time 128502.63ms, mfu 14.70%



| Repo | Steps  | Perplexity |
| ---- | ------ | ---------- |
| kchou0907/nanoGPT | 5000 |       |
|                   |      |       |

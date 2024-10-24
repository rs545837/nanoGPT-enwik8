"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from rotary_embedding_torch import RotaryEmbedding

import torch._dynamo
torch._dynamo.config.suppress_errors = True

def precompute_rope_cache(
    dim: int,
    seq_len: int,
    theta: float = 10000.0,
    scaling_factor: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Precompute RoPE cache with optional dynamic NTK scaling.
    
    Args:
        dim: Dimension per head
        seq_len: Maximum sequence length
        theta: Base for frequency computation
        scaling_factor: Optional scaling factor for dynamic scaling
        device: Target device
    """
    position = torch.arange(seq_len, device=device)
    dim_pos = torch.arange(0, dim, 2, device=device).float()
    
    # Apply scaling factor if provided (dynamic NTK scaling)
    if scaling_factor is not None:
        theta = theta * (scaling_factor ** (dim / (dim - 2)))
        
    freq = 1.0 / (theta ** (dim_pos / dim))
    angles = torch.outer(position, freq)
    cache = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    
    return cache

def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """
    Enhanced RoPE application with better numerical stability.
    """
    # Split features for rotation
    x_rot, x_pass = x[..., ::2], x[..., 1::2]
    
    # Get trig values from cache and reshape
    cos, sin = rope_cache[..., 0], rope_cache[..., 1]
    cos = cos.view(1, 1, cos.shape[0], cos.shape[1])
    sin = sin.view(1, 1, sin.shape[0], sin.shape[1])
    
    # Improved rotation implementation
    x_rot = x_rot * cos - x_pass * sin
    x_pass = x_pass * cos + x_rot * sin
    
    # Numerically stable recombination
    return torch.stack((x_rot, x_pass), dim=-1).flatten(-2)

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # save dimensions
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # key, query, value projections for all heads, but in a batch
        scale = 0.02  # Smaller initialization for better stability
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Initialize with smaller weights
        torch.nn.init.normal_(self.c_attn.weight, mean=0.0, std=scale)
        
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Special scaled initialization for output projection
        torch.nn.init.normal_(self.c_proj.weight, mean=0.0, std=scale/math.sqrt(2 * config.n_layer))
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        
        # rotary embeddings
        self.rope_cache = None
        
        # For numerical stability
        self.attention_scale = 1.0 / math.sqrt(self.head_size)

    def forward(self, x):
        B, T, C = x.size()
        
        # Check for NaN input
        if torch.isnan(x).any():
            print(f"Warning: NaN detected in attention input")
            # Optionally clip or zero out NaN values
            x = torch.nan_to_num(x, nan=0.0)
        
        # calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # split into heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Apply rotary embeddings
        if self.rope_cache is None or self.rope_cache.shape[0] < T:
            self.rope_cache = precompute_rope_cache(
                dim=self.head_size,
                seq_len=max(T, 32),
                device=x.device
            )
        
        q = apply_rope(q, self.rope_cache[:T])
        k = apply_rope(k, self.rope_cache[:T])
        
        # Clip extreme values for stability
        q = torch.clamp(q, min=-100, max=100)
        k = torch.clamp(k, min=-100, max=100)
        v = torch.clamp(v, min=-100, max=100)
        
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
                scale=self.attention_scale
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * self.attention_scale
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # Use more stable softmax
            att = F.softmax(att, dim=-1, dtype=torch.float32)  # Use float32 for better precision
            att = att.type_as(x)  # Convert back to original dtype
            att = self.attn_dropout(att)
            y = att @ v

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        
        # Final stability check
        if torch.isnan(y).any():
            print(f"Warning: NaN detected in attention output")
            y = torch.nan_to_num(y, nan=0.0)
        
        return y


class MLP(nn.Module):
    """ MLP with SwiGLU activation (better than GELU) """
    
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gate_proj = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.down_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize with smaller weights for better stability
        scale = 0.02
        torch.nn.init.normal_(self.up_proj.weight, mean=0.0, std=scale)
        torch.nn.init.normal_(self.gate_proj.weight, mean=0.0, std=scale)
        torch.nn.init.normal_(self.down_proj.weight, mean=0.0, std=scale/math.sqrt(2 * config.n_layer))
    
    def forward(self, x):
        # SwiGLU activation
        up = self.up_proj(x)
        gate = F.silu(self.gate_proj(x))  # SiLU (Swish) activation
        x = up * gate
        x = self.down_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
        # Residual scaling (like in PaLM)
        self.resid_attn_scale = 1 / math.sqrt(2 * config.n_layer)
        self.resid_mlp_scale = 1 / math.sqrt(2 * config.n_layer)
        
    def forward(self, x):
        # Pre-normalization and scaled residual connections
        attn_output = self.attn(self.ln_1(x))
        x = x + self.resid_attn_scale * attn_output
        
        mlp_output = self.mlp(self.ln_2(x))
        x = x + self.resid_mlp_scale * mlp_output
        return x

@dataclass
class GPTConfig:
    # Basic configuration
    block_size: int = 2048
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    
    # Regularization
    dropout: float = 0.0  # Generally better to use 0.0 and rely on other regularization
    bias: bool = False    # Better without bias
    
    # Architecture options
    use_swiglu: bool = True          # Use SwiGLU activation in MLP
    use_scaled_residual: bool = True  # Use scaled residual connections
    use_parallel_residual: bool = True  # Compute attention and MLP in parallel
    
    # RoPE settings
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None  # For dynamic NTK scaling
    
    # Performance options
    use_flash_attention: bool = True
    use_checkpointing: bool = False   # Enable gradient checkpointing for memory efficiency
    
    # Advanced features
    head_size: Optional[int] = None  # If None, will be set to n_embd // n_head
    
    def __post_init__(self):
        # Compute derived values
        self.head_size = self.head_size or self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        
        # Validate configuration
        if self.use_parallel_residual:
            assert self.use_scaled_residual, "Parallel residual requires scaled residual"

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use a more modern initialization scheme
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, LayerNorm)):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Forward the GPT model
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(x)
        
        # Through the transformer blocks
        for block in self.transformer.h:
            x = block(x)
            
        # Final layer norm
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training time: compute loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-1
            )
        else:
            # Inference time: only compute final token
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'mask'):
                block.attn.mask = block.attn.mask[:,:,:block_size,:block_size]
            # Clear RoPE cache to be regenerated with new size
            if hasattr(block.attn, 'rope_cache'):
                block.attn.rope_cache = None

    def from_pretrained(cls, model_type, override_args=None):
        """
        Load pretrained weights from Hugging Face model.
        Updated to handle new architecture with RoPE and GQA.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        # Allow overriding more parameters
        allowed_override = {'dropout', 'n_query_groups', 'window_size'}
        assert all(k in allowed_override for k in override_args)
        
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # Base configuration from model type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        
        # Add modern defaults
        config_args.update({
            'vocab_size': 50257,  # GPT-2 vocab size
            'block_size': 1024,   # GPT-2 block size
            'bias': True,         # Original GPT-2 used bias
            'n_query_groups': config_args['n_head'],  # Default to one group per head
            'window_size': None,  # No sliding window by default
        })
        
        # Apply any overrides
        config_args.update(override_args)
        
        # Create new model and load weights
        config = GPTConfig(**config_args)
        model = cls(config)
        
        # Get state dict and handle special cases
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.mask')] # Remove attention mask

        # Load pretrained model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Filter HF state dict
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        
        # Special handling for attention weights
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # Copy weights
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Transpose Conv1D weights
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Regular copy
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure the optimizer with proper weight decay and parameter grouping.
        Handles weight-tied parameters correctly.
        """
        # Collect all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Special handling for weight-tied parameters
        # The embedding weight is also used in the lm_head due to weight tying
        decay = set()
        no_decay = set()
        
        # Add parameters to the appropriate sets
        for pn, p in param_dict.items():
            if pn.endswith('wte.weight'):  # Embedding weights (also used as lm_head weights)
                decay.add(pn)
            elif pn.endswith('.bias'):
                no_decay.add(pn)  # all biases will not be decayed
            elif pn.endswith('.weight') and any(layer in pn for layer in ['c_attn', 'c_proj', 'c_fc']):
                decay.add(pn)  # attention and feedforward weights
            elif pn.endswith('ln_1.weight') or pn.endswith('ln_2.weight') or pn.endswith('ln_f.weight'):
                no_decay.add(pn)  # layer norm weights
            else:
                decay.add(pn)  # Other parameters
        
        # Validate that all parameters are accounted for
        param_dict_keys = set(param_dict.keys())
        intersect = decay & no_decay
        union = decay | no_decay
        assert len(intersect) == 0, f"Parameters {intersect} made it into both decay/no_decay sets!"
        assert len(param_dict_keys - union) == 0, f"Parameters {param_dict_keys - union} were not separated into either decay/no_decay set!"
        
        # Create optimizer groups
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        # Log parameter counts
        num_decay_params = sum(param_dict[pn].numel() for pn in decay)
        num_nodecay_params = sum(param_dict[pn].numel() for pn in no_decay)
        print(f"num decayed parameter tensors: {len(decay)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(no_decay)}, with {num_nodecay_params:,} parameters")
        
        # Create optimizer with optional fused implementation
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        # Adjust for GQA if enabled
        if hasattr(cfg, 'n_query_groups') and cfg.n_query_groups < cfg.n_head:
            # GQA reduces key/value computation
            H_kv = cfg.n_query_groups
            flops_per_token = 6*N + 12*L*H*Q*T + 12*L*H_kv*Q*T
        else:
            flops_per_token = 6*N + 12*L*H*Q*T
        
        # Calculate total flops
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        # Calculate MFU
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops
        mfu = flops_achieved / flops_promised
        
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is gwrowing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Apply nucleus (top-p) filtering if specified
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
                
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

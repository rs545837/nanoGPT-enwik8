import torch
import numpy as np
import argparse
import pickle
import math
from model import GPTConfig, GPT

def evaluate(model, data_loader, device):
    """
    Evaluate the model on the given data loader.
    Returns loss and bits per character (BPC).
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            
            with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                logits, loss = model(x, y)
                # For character-level models, we use the raw loss
                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()
    
    # Calculate metrics
    avg_loss = total_loss / total_tokens
    bpc = avg_loss / math.log(2)  # Convert nats to bits
    return avg_loss, bpc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model and data parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--dataset', type=str, default='enwik8_char', help='Dataset name')
    parser.add_argument('--out_dir', type=str, default='out', help='Output directory')
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--max_tokens', type=int, default=100000, help='Maximum number of tokens to evaluate on')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    # System parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Load model configuration
    config_dict = checkpoint['config']
    model_args = checkpoint['model_args']
    
    # Update vocab_size from dataset meta.pkl if available
    meta_path = f"data/{args.dataset}/meta.pkl"
    try:
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        model_args['vocab_size'] = meta['vocab_size']
        print(f"Using vocab_size={model_args['vocab_size']} from {meta_path}")
    except FileNotFoundError:
        print(f"No meta.pkl found at {meta_path}, using vocab_size from checkpoint")

    # Create model configuration
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # Load model weights
    state_dict = checkpoint['model']
    # Remove any DDP module prefix
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(args.device)
    
    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    print(f"Loaded model: {model_args['n_layer']} layers, {model_args['n_head']} heads, {model_args['n_embd']} embedding dim")

    # Prepare validation data
    print(f"Loading validation data from data/{args.dataset}/val.bin")
    val_data = np.memmap(f'data/{args.dataset}/val.bin', dtype=np.uint16, mode='r')
    
    # Limit the number of tokens if specified
    if args.max_tokens and args.max_tokens < len(val_data):
        val_data = val_data[:args.max_tokens]
    val_data = torch.from_numpy(val_data.astype(np.int64))
    
    # Create sequences
    block_size = model_args['block_size']
    n_sequences = (len(val_data) - 1) // block_size
    if n_sequences == 0:
        raise ValueError(f"Validation data too short for block_size {block_size}")
    
    x = torch.stack([val_data[i:i+block_size] for i in range(0, n_sequences * block_size, block_size)])
    y = torch.stack([val_data[i+1:i+block_size+1] for i in range(0, n_sequences * block_size, block_size)])
    
    # Create DataLoader
    val_dataset = torch.utils.data.TensorDataset(x, y)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Safer with memory mapping
        pin_memory=True if args.device == 'cuda' else False
    )

    # Evaluate
    print("\nStarting evaluation...")
    val_loss, bpc = evaluate(model, val_loader, args.device)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Average Loss (nats): {val_loss:.4f}")
    print(f"Bits per character (BPC): {bpc:.4f}")
    
    # Optional: save results
    results = {
        'loss': val_loss,
        'bpc': bpc,
        'n_tokens': n_sequences * block_size,
        'model_config': model_args,
    }
    
    print(f"\nProcessed {n_sequences * block_size:,} characters in {n_sequences:,} sequences")

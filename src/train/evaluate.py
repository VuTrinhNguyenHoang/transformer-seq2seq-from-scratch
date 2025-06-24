import os
import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from tqdm import tqdm

from src.model.transformer import Transformer
from src.data.prepare_data import get_dataloaders

def evaluate(args):
    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    model_args = checkpoint['model_args']
    
    # Create model with the same architecture
    model = Transformer(
        src_vocab_size=model_args['src_vocab_size'],
        tgt_vocab_size=model_args['tgt_vocab_size'],
        d_model=model_args['d_model'],
        num_heads=model_args['num_heads'],
        num_layers=model_args['num_layers'],
        d_ff=model_args['d_ff'],
        max_seq_length=model_args['max_seq_length'],
        dropout=0.0  # No dropout during evaluation
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    model.to(device)
    
    # Load data stats
    with open(os.path.join(args.data_dir, 'stats.json'), 'r') as f:
        stats = json.load(f)
    
    block_size = stats['max_seq_length']
    
    # Create dataloaders
    _, _, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        block_size=block_size,
        batch_size=args.batch_size
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Evaluation
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x, y[:, :-1])
            
            # Compute loss
            loss = criterion(logits.contiguous().view(-1, model_args['tgt_vocab_size']), y[:, 1:].contiguous().view(-1))
            total_loss += loss.item() * y[:, 1:].numel()
            
            # Count correct predictions
            predictions = torch.argmax(logits, dim=-1)
            mask = y[:, 1:] != 0
            correct_predictions += ((predictions == y[:, 1:]) & mask).sum().item()
            
            # Count tokens
            total_tokens += mask.sum().item()
    
    # Compute perplexity and accuracy
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    accuracy = correct_predictions / total_tokens
    
    # Print evaluation results
    print(f"Evaluation results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a GPT-mini model")
    
    # Data arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with processed data")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    
    args = parser.parse_args()
    
    evaluate(args)

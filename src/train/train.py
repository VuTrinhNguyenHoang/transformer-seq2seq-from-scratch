import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import argparse

from src.model.transformer import Transformer
from src.data.prepare_data import get_dataloaders

def train(args):
    # Load data stats to get vocabulary size
    with open(os.path.join(args.data_dir, 'stats.json'), 'r') as f:
        stats = json.load(f)
    
    src_vocab_size = stats['src_vocab_size']
    tgt_vocab_size = stats['tgt_vocab_size']
    max_seq_length = stats['max_seq_length']
    
    # Create model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_length=max_seq_length,
        dropout=args.dropout
    )
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    model.to(device)
    
    # Count and display number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{'='*50}")
    print(f"Model: GPT-mini")
    print(f"Vocabulary size: {src_vocab_size}")
    print(f"Hidden dimension: {args.d_model}")
    print(f"Number of layers: {args.num_layers}")
    print(f"Number of heads: {args.num_heads}")
    print(f"Device: {device}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*50}")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.98), 
        eps=1e-9, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=args.lr * 0.1
    )
    
    # Create dataloaders
    train_loader, dev_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        block_size=max_seq_length,
        batch_size=args.batch_size
    )

    if args.mode == 'dev':
        train_loader = dev_loader
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = model(x, y[:, :-1])
            
            # Compute loss
            loss = criterion(logits.contiguous().view(-1, tgt_vocab_size), y[:, 1:].contiguous().view(-1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            train_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": train_loss / train_steps})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                x, y = x.to(device), y.to(device)
                
                # Forward pass
                logits = model(x, y[:, :-1])
                
                # Compute loss
                loss = criterion(logits.contiguous().view(-1, tgt_vocab_size), y[:, 1:].contiguous().view(-1))
                
                # Update statistics
                val_loss += loss.item()
                val_steps += 1
        
        # Compute average losses
        avg_train_loss = train_loss / train_steps
        avg_val_loss = val_loss / val_steps
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{args.epochs} - Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}")
        
        # Update learning rate
        lr_scheduler.step()
        
        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Create checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'model_args': {
                    'src_vocab_size': src_vocab_size,
                    'tgt_vocab_size': tgt_vocab_size,
                    'd_model': args.d_model,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'd_ff': args.d_ff,
                    'max_seq_length': max_seq_length,
                    'dropout': args.dropout
                },
                'tokenizer_name': stats['tokenizer'],
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            }
            
            # Save checkpoint
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"Saved best model checkpoint to {os.path.join(args.output_dir, 'best_model.pt')}")
    
    # Save final model
    checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'model_args': {
            'src_vocab_size': src_vocab_size,
            'tgt_vocab_size': tgt_vocab_size,
            'd_model': args.d_model,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'd_ff': args.d_ff,
            'max_seq_length': max_seq_length,
            'dropout': args.dropout
        },
        'tokenizer_name': stats['tokenizer'],
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    }
    
    # Save final model
    torch.save(checkpoint, os.path.join(args.output_dir, 'final_model.pt'))
    print(f"Saved final model to {os.path.join(args.output_dir, 'final_model.pt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT-mini model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with processed data")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save model checkpoints")
    
    # Model arguments
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--mode", type=str, choices=["train", "dev"], default="train",
                        help="Training mode: 'train' for full training, 'dev' for debugging on dev set")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    
    args = parser.parse_args()
    
    train(args)

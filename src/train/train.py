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
    
    vocab_size = stats['vocab_size']
    
    # Create model
    model = Transformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_length=args.max_seq_length,
        dropout=args.dropout
    )
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    model.to(device)
    
    # Count and display number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{'='*50}")
    print(f"Model: Transformer Machine Translation EN-VI")
    print(f"Vocabulary size: {vocab_size}")
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
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )

    if args.mode == 'dev':
        train_loader = val_loader
    
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
        for src, tgt in progress_bar:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()

            # Forward pass
            logits = model(src, tgt[:, :-1])
            
            # Compute loss
            loss = criterion(logits.view(-1, vocab_size), tgt[:, 1:].contiguous().view(-1))
            
            # Backward pass and optimize
            loss.backward()
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
             for src, tgt in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                src, tgt = src.to(device), tgt.to(device)
                
                # Forward pass
                logits = model(src, tgt[:, :-1])
                
                # Compute loss
                loss = criterion(logits.view(-1, vocab_size), tgt[:, 1:].contiguous().view(-1))
                
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
            os.makedirs(args.output_dir, exist_ok=True)
            # Create checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'model_args': {
                    'vocab_size': vocab_size,
                    'd_model': args.d_model,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'd_ff': args.d_ff,
                    'max_seq_length': args.max_seq_length,
                    'dropout': args.dropout
                },
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            }
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
            'vocab_size': vocab_size,
            'd_model': args.d_model,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'd_ff': args.d_ff,
            'max_seq_length': args.max_seq_length,
            'dropout': args.dropout
        },
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    }
    torch.save(checkpoint, os.path.join(args.output_dir, 'final_model.pt'))
    print(f"Saved final model to {os.path.join(args.output_dir, 'final_model.pt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Machine Translation EN-VI")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with processed data")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save model checkpoints")
    
    # Model arguments
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=512, help="Feed-forward dimension")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--mode", type=str, choices=["train", "dev"], default="train")
    parser.add_argument("--no_cuda", action="store_true")
    
    args = parser.parse_args()
    
    train(args)
